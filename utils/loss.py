"""Main loss computation combining all loss components.

ComputeLoss orchestrates:
- Task-Aligned Assignment
- Classification Loss (BCE)
- Box Regression Loss (CIoU)
- Distribution Focal Loss (DFL)
"""

import torch
import torch.nn as nn
from typing import Tuple, Union, List

from .boxes import wh2xy, make_anchors
from .iou import IoUCalculator
from .assigner import TaskAlignedAssigner
from .losses import ClassificationLoss, BoxLoss, DFLoss


class ComputeLoss:
    """YOLO loss computation with Task-Aligned Assigner.

    Combines classification, box regression, and DFL losses
    with dynamic label assignment for optimal training.

    Components:
        - TaskAlignedAssigner: Dynamic GT-to-prediction assignment
        - ClassificationLoss: BCE for class predictions
        - BoxLoss: CIoU for box regression
        - DFLoss: Distribution Focal Loss for fine-grained regression
    """

    def __init__(self, model: nn.Module, params: dict):
        """Initialize loss computation.

        Args:
            model: YOLO model (or DDP wrapped model)
            params: Training parameters with keys:
                - 'cls': Classification loss weight
                - 'box': Box loss weight
                - 'dfl': DFL loss weight
        """
        # Handle DDP wrapped models
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device
        head = model.head

        # Model info
        self.stride = head.stride
        self.nc = head.nc  # number of classes
        self.no = head.no  # number of outputs per anchor
        self.device = device
        self.params = params

        # DFL parameters
        self.dfl_ch = head.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

        # Initialize components
        self.assigner = TaskAlignedAssigner(
            top_k=10,
            alpha=0.5,
            beta=6.0
        )
        self.cls_loss = ClassificationLoss()
        self.box_loss = BoxLoss(iou_type='ciou')
        self.dfl_loss = DFLoss()
        self.iou_calculator = IoUCalculator()

    def __call__(self, outputs: Union[Tuple, List],
                 targets: torch.Tensor) -> torch.Tensor:
        """Compute total loss.

        Args:
            outputs: Model predictions (list of feature maps or tuple)
            targets: Ground truth (N, 6) [batch_idx, cls, cx, cy, w, h]

        Returns:
            Total weighted loss
        """
        # Extract predictions
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        pred_output, pred_scores, anchor_points, stride_tensor = \
            self._prepare_predictions(x)

        # Prepare ground truth
        gt_labels, gt_bboxes, mask_gt = self._prepare_targets(
            targets, pred_scores, x
        )

        # Decode predicted boxes
        pred_bboxes = self._decode_boxes(pred_output, anchor_points)

        # Assign GT to predictions
        target_bboxes, target_scores, fg_mask = self.assigner.assign(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            gt_labels, gt_bboxes, mask_gt,
            anchor_points * stride_tensor,
            self.nc
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1.0)

        # Compute losses
        loss_cls = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum)
        loss_box, loss_dfl = self._compute_box_losses(
            pred_bboxes, pred_output, target_bboxes, target_scores,
            fg_mask, anchor_points, target_scores_sum
        )

        # Apply weights
        loss_cls *= self.params['cls']
        loss_box *= self.params['box']
        loss_dfl *= self.params['dfl']

        return loss_cls + loss_box + loss_dfl

    def _prepare_predictions(self, x: List[torch.Tensor]) -> Tuple:
        """Prepare predictions from feature maps.

        Returns:
            Tuple of (pred_output, pred_scores, anchor_points, stride_tensor)
        """
        # Concatenate predictions from all scales
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)

        # Split into box and class predictions
        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)

        # Reshape to (B, num_anchors, channels)
        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        # Generate anchors
        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

        return pred_output, pred_scores, anchor_points, stride_tensor

    def _prepare_targets(self, targets: torch.Tensor,
                        pred_scores: torch.Tensor,
                        x: List[torch.Tensor]) -> Tuple:
        """Prepare ground truth targets.

        Returns:
            Tuple of (gt_labels, gt_bboxes, mask_gt)
        """
        # Get image size
        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype,
                           device=self.device)
        size = size * self.stride[0]

        # Handle empty targets
        if targets.shape[0] == 0:
            gt = torch.zeros(pred_scores.shape[0], 0, 5, device=self.device)
        else:
            # Group targets by image
            batch_idx = targets[:, 0]
            _, counts = batch_idx.unique(return_counts=True)
            gt = torch.zeros(pred_scores.shape[0], counts.max(), 5,
                            device=self.device)

            for j in range(pred_scores.shape[0]):
                matches = batch_idx == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets[matches, 1:]

            # Convert to xyxy format
            gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        # Split labels and boxes
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        return gt_labels, gt_bboxes, mask_gt

    def _decode_boxes(self, pred_output: torch.Tensor,
                     anchor_points: torch.Tensor) -> torch.Tensor:
        """Decode box predictions from distribution to coordinates.

        Args:
            pred_output: Raw box predictions (B, num_anchors, 4*dfl_ch)
            anchor_points: Anchor point coordinates

        Returns:
            Decoded boxes in xyxy format
        """
        b, a, c = pred_output.shape

        # Apply softmax to get distribution
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)

        # Convert distribution to distance
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        # Convert ltrb to xyxy
        left_top, right_bottom = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - left_top,
                                 anchor_points + right_bottom), -1)

        return pred_bboxes

    def _compute_cls_loss(self, pred_scores: torch.Tensor,
                         target_scores: torch.Tensor,
                         target_scores_sum: float) -> torch.Tensor:
        """Compute classification loss."""
        loss = self.cls_loss(pred_scores, target_scores.to(pred_scores.dtype))
        return loss.sum() / target_scores_sum

    def _compute_box_losses(self, pred_bboxes: torch.Tensor,
                           pred_output: torch.Tensor,
                           target_bboxes: torch.Tensor,
                           target_scores: torch.Tensor,
                           fg_mask: torch.Tensor,
                           anchor_points: torch.Tensor,
                           target_scores_sum: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute box and DFL losses.

        Returns:
            Tuple of (box_loss, dfl_loss)
        """
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)

        if fg_mask.sum() == 0:
            return loss_box, loss_dfl

        # Get foreground weights
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        # Box loss (1 - CIoU)
        iou = self.iou_calculator.compute_ciou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask]
        )
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        left_top, right_bottom = torch.split(target_bboxes, 2, -1)
        target_ltrb = torch.cat((anchor_points - left_top,
                                 right_bottom - anchor_points), -1)
        target_ltrb = target_ltrb.clamp(0, self.dfl_ch - 1.01)

        loss_dfl = self.dfl_loss(
            pred_output[fg_mask].view(-1, self.dfl_ch),
            target_ltrb[fg_mask]
        )
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_box, loss_dfl
