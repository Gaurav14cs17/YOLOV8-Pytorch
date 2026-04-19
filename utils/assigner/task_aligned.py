# task_aligned.py
"""Task-Aligned Assigner for object detection."""

import torch
from torch.nn.functional import one_hot


class TaskAlignedAssigner:
    """
    Task-aligned One-stage Object Detection assigner.
    
    This assigner uses alignment metrics to match predictions to ground truth
    based on both classification scores and localization quality.
    
    The assignment process consists of these steps:
        1. Compute IoU between predictions and ground truth
        2. Calculate alignment metric: score^alpha × IoU^beta
        3. Select top-k candidates per ground truth
        4. Filter candidates that are inside ground truth boxes
        5. Resolve conflicts when one anchor matches multiple GTs
        6. Extract target boxes, scores, and foreground mask
        7. Normalize target scores by alignment quality
    """
    
    def __init__(self, num_classes, top_k=10, alpha=0.5, beta=6.0, eps=1e-9):
        """
        Initialize the Task-Aligned Assigner.
        
        Args:
            num_classes: Number of object classes
            top_k: Number of top candidates to consider per GT (default: 10)
            alpha: Classification score weight in alignment metric (default: 0.5)
            beta: IoU weight in alignment metric (default: 6.0)
            eps: Small epsilon for numerical stability (default: 1e-9)
        """
        self.nc = num_classes
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.bs = 1
        self.num_max_boxes = 0
    
    # =========================================================================
    # STEP 0: Handle Empty Ground Truth
    # =========================================================================
    
    def _get_empty_targets(self, pred_scores, pred_bboxes, device):
        """
        Return empty targets when there are no ground truth boxes.
        
        Args:
            pred_scores: Predicted scores [batch, num_anchors, num_classes]
            pred_bboxes: Predicted bboxes [batch, num_anchors, 4]
            device: Target device for tensors
        
        Returns:
            Tuple of empty tensors matching expected output shapes
        """
        return (
            torch.full_like(pred_scores[..., 0], self.nc).to(device),
            torch.zeros_like(pred_bboxes).to(device),
            torch.zeros_like(pred_scores).to(device),
            torch.zeros_like(pred_scores[..., 0]).to(device),
            torch.zeros_like(pred_scores[..., 0]).to(device)
        )
    
    # =========================================================================
    # STEP 1: Build Batch Index Tensors
    # =========================================================================
    
    def _build_batch_indices(self, true_labels):
        """
        Build index tensors for advanced indexing into pred_scores.
        
        Creates a 2D index tensor where:
            - i[0] contains batch indices (which image in batch)
            - i[1] contains class indices (from GT labels)
        
        This enables: pred_scores[i[0], :, i[1]] to get class scores
        for each (batch, GT) pair.
        
        Args:
            true_labels: Ground truth class labels [batch, num_boxes, 1]
        
        Returns:
            i: Index tensor [2, batch_size, num_max_boxes]
               i[0] = batch indices repeated for each GT
               i[1] = class indices from GT labels
        
        Example:
            batch_size=2, num_max_boxes=3, labels=[[0,1,2], [1,0,1]]
            i[0] = [[0,0,0], [1,1,1]]  # batch indices
            i[1] = [[0,1,2], [1,0,1]]  # class indices
        """
        # Initialize index tensor: [2, batch_size, num_max_boxes]
        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        
        # i[0]: Batch index for each (batch, GT) pair
        # torch.arange(bs) = [0, 1, ..., bs-1]
        # .view(-1, 1) reshapes to column: [[0], [1], ..., [bs-1]]
        # .repeat(1, num_max_boxes) tiles columns: [[0,0,0,...], [1,1,1,...], ...]
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        
        # i[1]: Class index from GT labels
        # true_labels shape: [batch, num_boxes, 1] -> squeeze to [batch, num_boxes]
        i[1] = true_labels.long().squeeze(-1)
        
        return i
    
    # =========================================================================
    # STEP 2: Compute Alignment Metric
    # =========================================================================
    
    def _compute_alignment_metric(self, pred_scores, pred_bboxes, true_bboxes, 
                                   batch_indices, iou_fn):
        """
        Compute the task-aligned metric for matching predictions to GTs.
        
        The alignment metric combines classification confidence and localization
        quality: metric = score^alpha × IoU^beta
        
        - alpha=0.5 (sqrt): Softens class scores to avoid overconfident predictions
        - beta=6.0: Strongly emphasizes localization - only high IoU survives
        
        Args:
            pred_scores: Predicted class scores [batch, num_anchors, num_classes]
            pred_bboxes: Predicted boxes [batch, num_anchors, 4]
            true_bboxes: Ground truth boxes [batch, num_boxes, 4]
            batch_indices: Index tensor from _build_batch_indices()
            iou_fn: Function to compute IoU (typically CIoU)
        
        Returns:
            overlaps: IoU values [batch, num_boxes, num_anchors]
            align_metric: Alignment scores [batch, num_boxes, num_anchors]
        """
        # Compute IoU between all GT boxes and all predicted boxes
        # true_bboxes: [B, N, 4] -> unsqueeze(2) -> [B, N, 1, 4]
        # pred_bboxes: [B, A, 4] -> unsqueeze(1) -> [B, 1, A, 4]
        # Result: [B, N, A, 1] -> squeeze -> [B, N, A]
        overlaps = iou_fn(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)  # Clamp negative IoU to 0
        
        # Extract class scores for each (batch, GT) pair
        # pred_scores[i[0], :, i[1]] gets scores for correct batch and class
        # Shape: [batch, num_boxes, num_anchors]
        class_scores = pred_scores[batch_indices[0], :, batch_indices[1]]
        
        # Compute alignment metric: score^alpha × IoU^beta
        align_metric = class_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return overlaps, align_metric
    
    # =========================================================================
    # STEP 3: Create Anchor-in-GT Mask
    # =========================================================================
    
    def _get_anchor_in_gt_mask(self, true_bboxes, anchors):
        """
        Create a mask indicating which anchors fall inside each GT box.
        
        An anchor is considered "inside" a GT box if its center point
        is within the box boundaries (all deltas are positive).
        
        Args:
            true_bboxes: Ground truth boxes [batch, num_boxes, 4] in xyxy format
            anchors: Anchor center points [num_anchors, 2]
        
        Returns:
            mask_in_gts: Boolean mask [batch, num_boxes, num_anchors]
                         True if anchor center is inside GT box
        
        Visualization:
            ┌─────────────────┐
            │     GT Box      │
            │  ●  ●  ●  ●     │  ● = anchor inside (True)
            │  ●  ●  ●  ●     │
            └─────────────────┘
              ○  ○  ○  ○  ○      ○ = anchor outside (False)
        """
        bs, n_boxes, _ = true_bboxes.shape
        
        # Split GT boxes into left-top (lt) and right-bottom (rb) corners
        # true_bboxes: [B, N, 4] -> view to [B*N, 1, 4] -> chunk to 2x [B*N, 1, 2]
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)
        # lt: [B*N, 1, 2] = (x1, y1)
        # rb: [B*N, 1, 2] = (x2, y2)
        
        # Compute deltas from anchor to GT box boundaries
        # anchors[None]: [1, A, 2] broadcasts with [B*N, 1, 2]
        # delta_to_lt = anchor - lt  (should be positive if anchor > left-top)
        # delta_to_rb = rb - anchor  (should be positive if anchor < right-bottom)
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        # Shape: [B*N, A, 4] = (dx_lt, dy_lt, dx_rb, dy_rb)
        
        # Anchor is inside if ALL four deltas are positive
        # .amin(3) gets minimum delta, .gt_(1e-9) checks if positive
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        
        return mask_in_gts
    
    # =========================================================================
    # STEP 4: Select Top-K Candidates
    # =========================================================================
    
    def _select_topk_candidates(self, metrics, true_mask, mask_in_gts):
        """
        Select top-k anchor candidates for each ground truth box.
        
        For each GT box, we select the k anchors with highest alignment
        metrics, then filter to keep only those inside the GT box.
        
        Args:
            metrics: Alignment metrics [batch, num_boxes, num_anchors]
            true_mask: Mask for valid GT boxes [batch, num_boxes, 1]
            mask_in_gts: Anchors inside GT mask [batch, num_boxes, num_anchors]
        
        Returns:
            mask_pos: Positive assignment mask [batch, num_boxes, num_anchors]
                      True for anchors assigned to each GT
        
        Process:
            1. Select top-k anchors by metric for each GT
            2. Convert indices to one-hot representation
            3. Filter duplicates (anchor in top-k of multiple GTs)
            4. Combine with mask_in_gts to get final positives
        """
        num_anchors = metrics.shape[-1]
        
        # Create mask for valid top-k slots
        # true_mask: [B, N, 1] -> repeat to [B, N, k]
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        
        # Select top-k anchors by metric for each GT
        # top_k_indices: [B, N, k] - indices of top-k anchors per GT
        top_k_metrics, top_k_indices = torch.topk(
            metrics, self.top_k, dim=-1, largest=True
        )
        
        # Handle case where top_k_mask is None
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        
        # Zero out indices for invalid GT slots
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        
        # Convert top-k indices to binary mask using one-hot encoding
        # one_hot: [B, N, k] -> [B, N, k, A]
        # sum(-2): [B, N, A] - how many times each anchor appears in top-k
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        
        # Filter anchors that appear in top-k of multiple GTs (set to 0)
        # This prevents one anchor from being assigned to multiple GTs
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        
        # Combine all masks: top-k AND inside GT AND valid GT
        # mask_pos[b, n, a] = True if anchor a is positive for GT n in batch b
        mask_pos = mask_top_k * mask_in_gts * true_mask
        
        return mask_pos
    
    # =========================================================================
    # STEP 5: Resolve Multi-GT Conflicts
    # =========================================================================
    
    def _resolve_multi_gt_conflicts(self, mask_pos, overlaps):
        """
        Resolve cases where one anchor is assigned to multiple GT boxes.
        
        When an anchor falls in the top-k of multiple GTs, we keep only
        the assignment to the GT with highest IoU overlap.
        
        Args:
            mask_pos: Current positive mask [batch, num_boxes, num_anchors]
            overlaps: IoU values [batch, num_boxes, num_anchors]
        
        Returns:
            mask_pos: Updated mask with conflicts resolved
            fg_mask: Foreground mask [batch, num_anchors] - True for positive anchors
        
        Example:
            Anchor 5 matches GT0 (IoU=0.7) and GT1 (IoU=0.5)
            → Keep only GT0 assignment (higher IoU)
        """
        # Count how many GTs each anchor is assigned to
        # fg_mask: [B, A] - sum over GT dimension
        fg_mask = mask_pos.sum(-2)
        
        # Check if any anchor is assigned to multiple GTs
        if fg_mask.max() > 1:
            # Find anchors with multiple GT assignments
            # mask_multi_gts: [B, N, A] - True where anchor has >1 GT
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            
            # For each anchor, find which GT has maximum IoU
            # max_overlaps_idx: [B, A] - index of GT with max overlap
            max_overlaps_idx = overlaps.argmax(1)
            
            # Create one-hot mask for the max-overlap GT
            # is_max_overlaps: [B, N, A] - True only for max IoU GT
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            
            # Replace multi-GT assignments with max-IoU only
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            
            # Recompute foreground mask
            fg_mask = mask_pos.sum(-2)
        
        return mask_pos, fg_mask
    
    # =========================================================================
    # STEP 6: Extract Target Assignments
    # =========================================================================
    
    def _get_targets(self, mask_pos, fg_mask, true_labels, true_bboxes):
        """
        Extract target boxes and scores for each anchor.
        
        For each positive anchor, look up which GT it's assigned to
        and extract that GT's box and class label.
        
        Args:
            mask_pos: Positive mask [batch, num_boxes, num_anchors]
            fg_mask: Foreground mask [batch, num_anchors]
            true_labels: GT class labels [batch, num_boxes, 1]
            true_bboxes: GT boxes [batch, num_boxes, 4]
        
        Returns:
            target_gt_idx: Index of assigned GT [batch, num_anchors]
            target_bboxes: Assigned GT boxes [batch, num_anchors, 4]
            target_scores: One-hot class labels [batch, num_anchors, num_classes]
        """
        # Find which GT each anchor is assigned to
        # target_gt_idx: [B, A] - local GT index per batch
        target_gt_idx = mask_pos.argmax(-2)
        
        # Convert local GT indices to global flat indices
        # This is needed because we'll flatten true_labels and true_bboxes
        # Global index = local_idx + batch_offset
        # batch_offset = batch_index × num_max_boxes
        batch_index = torch.arange(
            end=self.bs, dtype=torch.int64, device=true_labels.device
        )[..., None]  # [B, 1] for broadcasting
        
        # Add batch offset: [B, A] + [B, 1] broadcasts to [B, A]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        
        # Extract target labels using flat indexing
        # true_labels: [B, N, 1] -> flatten to [B*N]
        # target_gt_idx: [B, A] selects from flattened labels
        target_labels = true_labels.long().flatten()[target_gt_idx]
        
        # Extract target boxes using flat indexing
        # true_bboxes: [B, N, 4] -> view to [B*N, 4]
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]
        
        # Convert labels to one-hot scores
        target_labels.clamp(0)  # Ensure non-negative
        target_scores = one_hot(target_labels, self.nc)
        
        # Zero out scores for background anchors
        # fg_scores_mask: [B, A, NC] - True for foreground anchors
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        
        return target_gt_idx, target_bboxes, target_scores
    
    # =========================================================================
    # STEP 7: Normalize Target Scores
    # =========================================================================
    
    def _normalize_scores(self, target_scores, align_metric, overlaps, mask_pos):
        """
        Normalize target scores by alignment quality.
        
        Instead of hard one-hot labels, we use soft labels weighted by
        the alignment metric. This provides better training signal.
        
        Normalization formula:
            normalized_score = (metric × max_overlap) / max_metric
        
        Args:
            target_scores: One-hot class scores [batch, num_anchors, num_classes]
            align_metric: Alignment metrics [batch, num_boxes, num_anchors]
            overlaps: IoU values [batch, num_boxes, num_anchors]
            mask_pos: Positive mask [batch, num_boxes, num_anchors]
        
        Returns:
            target_scores: Normalized soft labels [batch, num_anchors, num_classes]
        
        Why normalize?
            - Hard labels (0 or 1) don't reflect assignment quality
            - Soft labels weight loss by how good the match is
            - Better match → higher weight → more gradient signal
        """
        # Mask alignment metric to only positive assignments
        align_metric = align_metric * mask_pos
        
        # Get maximum alignment metric per GT
        # pos_align_metrics: [B, N, 1]
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        
        # Get maximum overlap per GT (among positive anchors)
        # pos_overlaps: [B, N, 1]
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        
        # Compute normalized alignment metric
        # norm = (metric × max_overlap) / max_metric, then max over GTs
        # Shape: [B, N, A] -> amax(-2) -> [B, A] -> unsqueeze -> [B, A, 1]
        norm_align_metric = (
            align_metric * pos_overlaps / (pos_align_metrics + self.eps)
        ).amax(-2).unsqueeze(-1)
        
        # Apply normalization to target scores
        # target_scores: [B, A, NC] × [B, A, 1] broadcasts correctly
        target_scores = target_scores * norm_align_metric
        
        return target_scores
    
    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, true_labels, true_bboxes, 
                 true_mask, anchors, iou_fn):
        """
        Perform task-aligned assignment.
        
        This is the main entry point that orchestrates all assignment steps.
        
        Args:
            pred_scores: Predicted scores [batch, num_anchors, num_classes]
            pred_bboxes: Predicted boxes [batch, num_anchors, 4]
            true_labels: GT class labels [batch, num_boxes, 1]
            true_bboxes: GT boxes [batch, num_boxes, 4]
            true_mask: Valid GT mask [batch, num_boxes, 1]
            anchors: Anchor points [num_anchors, 2]
            iou_fn: IoU computation function
        
        Returns:
            target_bboxes: Assigned GT boxes [batch, num_anchors, 4]
            target_scores: Soft class labels [batch, num_anchors, num_classes]
            fg_mask: Foreground mask [batch, num_anchors] (boolean)
        """
        # Store batch info
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)
        
        # Step 0: Handle empty ground truth
        if self.num_max_boxes == 0:
            return self._get_empty_targets(pred_scores, pred_bboxes, true_bboxes.device)
        
        # Step 1: Build batch index tensors for advanced indexing
        batch_indices = self._build_batch_indices(true_labels)
        
        # Step 2: Compute alignment metric (score^α × IoU^β)
        overlaps, align_metric = self._compute_alignment_metric(
            pred_scores, pred_bboxes, true_bboxes, batch_indices, iou_fn
        )
        
        # Step 3: Create mask for anchors inside GT boxes
        mask_in_gts = self._get_anchor_in_gt_mask(true_bboxes, anchors)
        
        # Step 4: Select top-k candidates per GT
        # metrics = alignment_metric × inside_gt_mask
        metrics = align_metric * mask_in_gts
        mask_pos = self._select_topk_candidates(metrics, true_mask, mask_in_gts)
        
        # Step 5: Resolve conflicts (one anchor → multiple GTs)
        mask_pos, fg_mask = self._resolve_multi_gt_conflicts(mask_pos, overlaps)
        
        # Step 6: Extract target boxes and scores
        target_gt_idx, target_bboxes, target_scores = self._get_targets(
            mask_pos, fg_mask, true_labels, true_bboxes
        )
        
        # Step 7: Normalize scores by alignment quality
        target_scores = self._normalize_scores(
            target_scores, align_metric, overlaps, mask_pos
        )
        
        return target_bboxes, target_scores, fg_mask.bool()
