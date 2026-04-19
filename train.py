"""
YOLOv8 Training Script

Clean, modular training implementation with:
- Configurable training parameters
- Multi-GPU support (DDP)
- Mixed precision training
- Exponential Moving Average (EMA)
- Warmup scheduling
- Comprehensive evaluation
- Epoch visualization (saves sample detections)
"""

import argparse
import copy
import csv
import os
import shutil
import warnings
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import distributed as data_distributed

from model import yolo_v8_n, yolo_v8_s, yolo_v8_m, yolo_v8_l, yolo_v8_x
from dataloader import Dataset
from utils import (
    ComputeLoss, EMA, AverageMeter,
    setup_seed, setup_multi_processes,
    strip_optimizer, clip_gradients,
    non_max_suppression, scale, box_iou, compute_ap
)

warnings.filterwarnings("ignore")

# Supported image formats
IMAGE_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


# =============================================================================
# Output Manager
# =============================================================================

class OutputManager:
    """Manages all training outputs in a single folder structure."""
    
    def __init__(self, base_dir: str = 'output'):
        self.base_dir = Path(base_dir)
        self.weights_dir = self.base_dir / 'weights'
        self.samples_dir = self.base_dir / 'epoch_samples'
        self.metrics_dir = self.base_dir / 'metrics'
        
        # Create all directories
        for d in [self.weights_dir, self.samples_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_weight_path(self, name: str) -> Path:
        return self.weights_dir / name
    
    def get_metrics_path(self, name: str) -> Path:
        return self.metrics_dir / name


# =============================================================================
# Epoch Visualization
# =============================================================================

class EpochVisualizer:
    """Saves sample detection images at each epoch and manages folder cleanup."""
    
    def __init__(self, output_dir: Path, max_folders: int = 5):
        self.output_dir = output_dir
        self.max_folders = max_folders
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for visualization (BGR format)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3)).tolist()
    
    def save_sample(self, model: nn.Module, dataloader: DataLoader, 
                   epoch: int, device: torch.device, class_names: Dict[int, str]):
        """Save one sample detection image for the current epoch."""
        # Create epoch folder
        epoch_folder = self.output_dir / f'epoch_{epoch:04d}'
        epoch_folder.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        
        # Get one batch
        for images, targets, shapes in dataloader:
            images = images.to(device).float() / 255.0
            
            with torch.no_grad():
                # Run inference
                outputs = model(images)
                detections = non_max_suppression(outputs, conf_threshold=0.25, iou_threshold=0.45)
            
            # Process first image in batch
            img_tensor = images[0]
            det = detections[0] if detections else torch.zeros((0, 6))
            shape = shapes[0] if shapes else None
            
            # Convert tensor to numpy image (CHW -> HWC, RGB -> BGR)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Draw detections
            if len(det) > 0:
                for *xyxy, conf, cls_id in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_id = int(cls_id)
                    color = self.colors[cls_id % len(self.colors)]
                    
                    # Draw box
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f'{class_names.get(cls_id, f"cls_{cls_id}")} {conf:.2f}'
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_np, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(img_np, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add epoch info
            info_text = f'Epoch {epoch} | Detections: {len(det)}'
            cv2.putText(img_np, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save image
            save_path = epoch_folder / f'sample_epoch_{epoch:04d}.jpg'
            cv2.imwrite(str(save_path), img_np)
            
            break  # Only process first batch
        
        model.train()
        
        # Cleanup old folders
        self._cleanup_old_folders()
        
        return str(epoch_folder)
    
    def _cleanup_old_folders(self):
        """Keep only the last N epoch folders."""
        # Get all epoch folders sorted by epoch number
        epoch_folders = sorted(
            [f for f in self.output_dir.iterdir() if f.is_dir() and f.name.startswith('epoch_')],
            key=lambda x: int(x.name.split('_')[1])
        )
        
        # Delete oldest folders if we have more than max_folders
        while len(epoch_folders) > self.max_folders:
            oldest = epoch_folders.pop(0)
            shutil.rmtree(oldest)
            print(f'Deleted old epoch folder: {oldest.name}')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Training configuration container."""
    
    def __init__(self, config_path: str = 'config/config.yml'):
        with open(config_path, errors='ignore') as f:
            self.params = yaml.safe_load(f)
    
    def __getitem__(self, key):
        return self.params[key]
    
    def get(self, key, default=None):
        return self.params.get(key, default)
    
    @property
    def num_classes(self) -> int:
        return len(self.params['names'])
    
    @property
    def class_names(self) -> Dict[int, str]:
        return self.params['names']


# =============================================================================
# Data Loading
# =============================================================================

class DataManager:
    """Handles dataset loading and DataLoader creation."""
    
    def __init__(self, config: Config, args):
        self.config = config
        self.args = args
    
    @staticmethod
    def list_images(root: str) -> List[str]:
        """List all image files in directory."""
        root = Path(root)
        images = []
        for ext in IMAGE_FORMATS:
            images.extend([str(p) for p in root.rglob(f'*.{ext}')])
            images.extend([str(p) for p in root.rglob(f'*.{ext.upper()}')])
        return sorted(images)
    
    def get_image_paths(self, split: str = 'train') -> List[str]:
        """Get filtered image paths with corresponding labels."""
        data_cfg = self.config.get('data', {})
        img_dir = data_cfg.get(split, f'dataset/images/{split}')
        label_root = Path(img_dir.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'))
        
        all_images = self.list_images(img_dir)
        valid_images = []
        
        for img_path in all_images:
            stem = Path(img_path).stem
            label_path = label_root / f'{stem}.txt'
            if label_path.exists():
                valid_images.append(img_path)
        
        if not valid_images:
            raise ValueError(
                f"No images found for split '{split}'.\n"
                f"Searched: {img_dir}\n"
                f"Ensure labels exist in: {label_root}"
            )
        
        return valid_images
    
    def create_dataloader(self, split: str = 'train', 
                         batch_size: Optional[int] = None) -> Tuple[DataLoader, Optional[data_distributed.DistributedSampler]]:
        """Create DataLoader for specified split."""
        is_train = (split == 'train')
        batch_size = batch_size or self.args.batch_size
        
        image_paths = self.get_image_paths(split)
        print(f"Loaded {len(image_paths)} {split} samples")
        
        dataset = Dataset(
            image_paths, 
            self.args.input_size, 
            self.config.params, 
            augment=is_train
        )
        
        sampler = None
        if self.args.world_size > 1 and is_train:
            sampler = data_distributed.DistributedSampler(dataset)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and is_train),
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )
        
        return loader, sampler


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Factory for creating YOLOv8 model variants."""
    
    VARIANTS = {
        'nano': yolo_v8_n,
        'n': yolo_v8_n,
        'small': yolo_v8_s,
        's': yolo_v8_s,
        'medium': yolo_v8_m,
        'm': yolo_v8_m,
        'large': yolo_v8_l,
        'l': yolo_v8_l,
        'xlarge': yolo_v8_x,
        'x': yolo_v8_x,
    }
    
    @classmethod
    def create(cls, variant: str, num_classes: int) -> nn.Module:
        """Create model variant."""
        variant = variant.lower()
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(cls.VARIANTS.keys())}")
        return cls.VARIANTS[variant](num_classes)


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

class OptimizerFactory:
    """Creates and configures optimizer with parameter groups."""
    
    @staticmethod
    def create(model: nn.Module, config: Config, args) -> torch.optim.Optimizer:
        """Create optimizer with separate parameter groups."""
        biases, bn_weights, other_weights = [], [], []
        
        for module in model.modules():
            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                biases.append(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                bn_weights.append(module.weight)
            elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                other_weights.append(module.weight)
        
        # Calculate effective weight decay
        accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
        weight_decay = config['weight_decay'] * args.batch_size * args.world_size * accumulate / 64
        
        optimizer = torch.optim.SGD(
            biases,
            lr=config['lr0'],
            momentum=config['momentum'],
            nesterov=True
        )
        optimizer.add_param_group({'params': other_weights, 'weight_decay': weight_decay})
        optimizer.add_param_group({'params': bn_weights})
        
        return optimizer


class SchedulerFactory:
    """Creates learning rate scheduler."""
    
    @staticmethod
    def create(optimizer: torch.optim.Optimizer, config: Config, epochs: int):
        """Create cosine annealing scheduler."""
        def lr_lambda(epoch):
            return (1 - epoch / epochs) * (1.0 - config['lrf']) + config['lrf']
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Engine
# =============================================================================

class Trainer:
    """Main training engine."""
    
    def __init__(self, args, config: Config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_main = (args.local_rank == 0)
        
        # Output manager (single folder for all outputs)
        self.output = OutputManager('output') if self.is_main else None
        
        # Initialize components
        self._init_model()
        self._init_optimizer()
        self._init_data()
        self._init_training()
        
        # Epoch visualization (saves sample images, keeps last 5 folders)
        self.visualizer = EpochVisualizer(
            output_dir=self.output.samples_dir,
            max_folders=5
        ) if self.is_main else None
        
        self.best_map = 0.0
    
    def _init_model(self):
        """Initialize model."""
        variant = self.config.get('variant', 'nano')
        self.model = ModelFactory.create(variant, self.config.num_classes)
        self.model.to(self.device)
        
        if self.args.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.accumulate = max(round(64 / (self.args.batch_size * self.args.world_size)), 1)
        self.optimizer = OptimizerFactory.create(self.model, self.config, self.args)
        self.scheduler = SchedulerFactory.create(self.optimizer, self.config, self.args.epochs)
    
    def _init_data(self):
        """Initialize data loaders."""
        self.data_manager = DataManager(self.config, self.args)
        self.train_loader, self.train_sampler = self.data_manager.create_dataloader('train')
        self.val_loader, _ = self.data_manager.create_dataloader('val', batch_size=4)
    
    def _init_training(self):
        """Initialize training components."""
        self.criterion = ComputeLoss(self.model, self.config.params)
        self.ema = EMA(self.model) if self.is_main else None
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.num_batches = len(self.train_loader)
        self.num_warmup = max(round(self.config['warmup_epochs'] * self.num_batches), 1000)
    
    def _warmup(self, iteration: int, epoch: int):
        """Apply warmup to learning rate and momentum."""
        if iteration > self.num_warmup:
            return
        
        warmup_ratio = np.interp(
            iteration, [0, self.num_warmup],
            [1, 64 / (self.args.batch_size * self.args.world_size)]
        )
        self.accumulate = max(1, round(warmup_ratio))
        
        lr_lambda = self.scheduler.lr_lambdas[0]
        for j, group in enumerate(self.optimizer.param_groups):
            if j == 0:  # biases
                target_lr = np.interp(
                    iteration, [0, self.num_warmup],
                    [self.config['warmup_bias_lr'], group['initial_lr'] * lr_lambda(epoch)]
                )
            else:
                target_lr = np.interp(
                    iteration, [0, self.num_warmup],
                    [0.0, group['initial_lr'] * lr_lambda(epoch)]
                )
            group['lr'] = target_lr
            
            if 'momentum' in group:
                group['momentum'] = np.interp(
                    iteration, [0, self.num_warmup],
                    [self.config['warmup_momentum'], self.config['momentum']]
                )
    
    def train_epoch(self, epoch: int) -> AverageMeter:
        """Train for one epoch."""
        self.model.train()
        
        # Disable mosaic for last 10 epochs
        if self.args.epochs - epoch <= 10:
            self.train_loader.dataset.mosaic = False
        
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        pbar = tqdm.tqdm(enumerate(self.train_loader), total=self.num_batches) if self.is_main else enumerate(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for i, (images, targets, _) in pbar:
            iteration = i + epoch * self.num_batches
            images = images.to(self.device).float() / 255.0
            targets = targets.to(self.device)
            
            self._warmup(iteration, epoch)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            loss_meter.update(loss.item(), images.size(0))
            scaled_loss = loss * self.args.batch_size * self.args.world_size
            
            # Backward pass
            self.scaler.scale(scaled_loss).backward()
            
            # Optimizer step
            if iteration % self.accumulate == 0:
                self.scaler.unscale_(self.optimizer)
                clip_gradients(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.ema:
                    self.ema.update(self.model)
            
            if self.is_main:
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                pbar.set_description(
                    f'Epoch [{epoch+1}/{self.args.epochs}] '
                    f'Memory: {mem:.3f}G Loss: {loss_meter.avg:.4g}'
                )
        
        self.scheduler.step()
        return loss_meter
    
    @torch.no_grad()
    def evaluate(self, model: Optional[nn.Module] = None) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        model = model or (self.ema.ema if self.ema else self.model)
        model.eval()
        model.half()
        
        iou_thresholds = torch.linspace(0.5, 0.95, 10).to(self.device)
        n_iou = iou_thresholds.numel()
        metrics = []
        
        pbar = tqdm.tqdm(self.val_loader, desc='Evaluating')
        
        for images, targets, shapes in pbar:
            images = images.to(self.device).half() / 255.0
            targets = targets.to(self.device)
            batch_size, _, height, width = images.shape
            
            outputs = model(images)
            targets[:, 2:] *= torch.tensor([width, height, width, height], device=self.device)
            
            outputs = non_max_suppression(outputs, conf_threshold=0.001, iou_threshold=0.65)
            
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=self.device)
                
                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, torch.tensor([]).to(self.device), 
                                      torch.tensor([]).to(self.device), labels[:, 0]))
                    continue
                
                detections = output.clone()
                scale(detections[:, :4], images[i].shape[1:], shapes[i][0], shapes[i][1])
                
                if labels.shape[0]:
                    tbox = self._convert_labels_to_xyxy(labels)
                    scale(tbox, images[i].shape[1:], shapes[i][0], shapes[i][1])
                    
                    correct = self._compute_correct(detections, labels, tbox, iou_thresholds)
                
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))
        
        map50, mean_ap = self._compute_metrics(metrics)
        model.float()
        return map50, mean_ap
    
    def _convert_labels_to_xyxy(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels from xywh to xyxy format."""
        tbox = labels[:, 1:5].clone()
        tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2
        tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2
        tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2
        tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2
        return tbox
    
    def _compute_correct(self, detections, labels, tbox, iou_thresholds):
        """Compute correct predictions matrix."""
        n_iou = iou_thresholds.numel()
        correct_np = np.zeros((detections.shape[0], n_iou), dtype=bool)
        
        t_tensor = torch.cat((labels[:, 0:1], tbox), dim=1)
        iou = box_iou(t_tensor[:, 1:], detections[:, :4])
        correct_class = t_tensor[:, 0:1] == detections[:, 5]
        
        for j in range(n_iou):
            matches = torch.where((iou >= iou_thresholds[j]) & correct_class)
            if matches[0].shape[0]:
                matched = torch.cat(
                    (torch.stack(matches, dim=1), iou[matches[0], matches[1]].unsqueeze(1)),
                    dim=1
                ).cpu().numpy()
                
                if matched.shape[0] > 1:
                    matched = matched[matched[:, 2].argsort()[::-1]]
                    matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
                    matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
                
                correct_np[matched[:, 1].astype(int), j] = True
        
        return torch.tensor(correct_np, dtype=torch.bool, device=detections.device)
    
    def _compute_metrics(self, metrics: List) -> Tuple[float, float]:
        """Compute mAP metrics from collected results."""
        if not metrics:
            return 0.0, 0.0
        
        metrics_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        
        if len(metrics_np) and metrics_np[0].any():
            _, _, _, _, map50, mean_ap = compute_ap(*metrics_np)
        else:
            map50, mean_ap = 0.0, 0.0
        
        print(f'mAP@0.5: {map50:.3g} mAP@0.5:0.95: {mean_ap:.3g}')
        return map50, mean_ap
    
    def save_checkpoint(self, epoch: int, map50: float, mean_ap: float):
        """Save model checkpoint."""
        if not self.is_main:
            return
        
        model_to_save = self.ema.ema if self.ema else self.model
        checkpoint = {'model': copy.deepcopy(model_to_save).half()}
        
        # Save last
        torch.save(checkpoint, self.output.get_weight_path('last.pt'))
        
        # Save best
        if mean_ap > self.best_map:
            self.best_map = mean_ap
            torch.save(checkpoint, self.output.get_weight_path('best.pt'))
    
    def train(self):
        """Main training loop."""
        metrics_file = self.output.get_metrics_path('step.csv') if self.is_main else Path('step.csv')
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mAP@50', 'mAP'])
            if self.is_main:
                writer.writeheader()
            
            for epoch in range(self.args.epochs):
                loss_meter = self.train_epoch(epoch)
                
                if self.is_main:
                    # Evaluate
                    map50, mean_ap = self.evaluate()
                    
                    # Save sample detection image for this epoch
                    if self.visualizer:
                        model_to_vis = self.ema.ema if self.ema else self.model
                        self.visualizer.save_sample(
                            model=model_to_vis,
                            dataloader=self.val_loader,
                            epoch=epoch + 1,
                            device=self.device,
                            class_names=self.config.class_names
                        )
                    
                    # Log metrics
                    writer.writerow({
                        'epoch': str(epoch + 1).zfill(3),
                        'mAP@50': f'{map50:.3f}',
                        'mAP': f'{mean_ap:.3f}'
                    })
                    f.flush()
                    self.save_checkpoint(epoch, map50, mean_ap)
        
        # Strip optimizer from saved weights
        if self.is_main:
            for name in ['best.pt', 'last.pt']:
                weight_file = self.output.get_weight_path(name)
                if weight_file.exists():
                    strip_optimizer(str(weight_file))
        
        torch.cuda.empty_cache()


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Training')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DDP')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Config file path')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run evaluation')
    return parser.parse_args()


def setup_distributed(args):
    """Setup distributed training environment."""
    args.local_rank = int(os.getenv('LOCAL_RANK', args.local_rank))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


def main():
    """Main entry point."""
    args = parse_args()
    setup_distributed(args)
    
    setup_seed()
    setup_multi_processes()
    
    config = Config(args.config)
    trainer = Trainer(args, config)
    
    if args.train:
        trainer.train()
    
    if args.test:
        trainer.evaluate()


if __name__ == '__main__':
    main()
