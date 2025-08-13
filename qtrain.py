import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import yaml
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from qmodel.qyolov8 import *
from nets import nn
from utils import util
from utils.dataset import Dataset
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class

warnings.filterwarnings("ignore")


def create_yolo_variant(variant_name, num_classes=80):
    """Factory function to create different YOLOv8 variants."""
    variants = {
        'nano': {'depth': [1, 2, 2], 'width': [3, 16, 32, 64, 128, 256]},
        'tiny': {'depth': [1, 2, 2], 'width': [3, 24, 48, 96, 192, 384]},
        'small': {'depth': [1, 2, 2], 'width': [3, 32, 64, 128, 256, 512]},
        'medium': {'depth': [2, 4, 4], 'width': [3, 48, 96, 192, 384, 576]},
        'large': {'depth': [3, 6, 6], 'width': [3, 64, 128, 256, 512, 512]},
        'xlarge': {'depth': [3, 6, 6], 'width': [3, 80, 160, 320, 640, 640]}
    }
    
    config = variants.get(variant_name.lower())
    if not config:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(variants.keys())}")
    
    return YOLOv8(config['width'], config['depth'], num_classes)


class Trainer:
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.init_model()
        
        # Setup distributed training if needed
        self.setup_distributed()
        
        # Initialize data loaders
        self.init_data_loaders()
        
        # Initialize optimizer and scheduler
        self.init_optimizer()
        
        # Loss function
        self.criterion = ComputeLoss(self.model, params)
        
        # Metrics
        self.best_ap = 0.0
        self.writer = None
        if self.is_main_process:
            self.writer = SummaryWriter(log_dir='runs/train')
            
    def init_model(self):
        """Initialize model with QAT preparation using factory function"""
        # Get model variant from params or default to 'nano'
        model_variant = self.params.get('variant', 'nano')
        self.model = create_yolo_variant(model_variant, len(self.params['names']))
        
        # Load pretrained weights if available
        weight_file = f'./weights/v8_{model_variant[0]}.pth'  # v8_n.pth, v8_s.pth etc.
        if os.path.exists(weight_file):
            state = torch.load(weight_file)['model']
            self.model.load_state_dict(state.float().state_dict())
        
        # Fuse Conv+BN layers for QAT
        for m in self.model.modules():
            if isinstance(m, nn.Conv) and hasattr(m, 'norm'):
                torch.ao.quantization.fuse_modules(m, [["conv", "norm"]], True)
        
        self.model.train()
        self.model = nn.QAT(self.model)
        self.model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare_qat(self.model, inplace=True)
        self.model.to(self.device)
    
    def setup_distributed(self):
        """Setup distributed training"""
        self.is_main_process = self.args.local_rank == 0
        
        if self.args.distributed:
            # DDP mode
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                module=self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
    
    def init_data_loaders(self):
        """Initialize train and validation data loaders"""
        # Train dataset
        with open('../Dataset/COCO/train2017.txt') as f:
            train_files = [f'../Dataset/COCO/images/train2017/{x.rstrip().split("/")[-1]}' 
                          for x in f.readlines()]
        
        train_sampler = None
        if self.args.distributed:
            train_sampler = data.distributed.DistributedSampler(train_files)
            
        self.train_loader = data.DataLoader(
            Dataset(train_files, self.args.input_size, self.params, True),
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )
        
        # Validation dataset
        with open('../Dataset/COCO/val2017.txt') as f:
            val_files = [f'../Dataset/COCO/images/val2017/{x.rstrip().split("/")[-1]}' 
                        for x in f.readlines()]
            
        self.val_loader = data.DataLoader(
            Dataset(val_files, self.args.input_size, self.params, False),
            batch_size=self.args.batch_size // 2,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=Dataset.collate_fn
        )
    
    def init_optimizer(self):
        """Initialize optimizer and scheduler"""
        # Adjust weight decay based on batch size
        self.accumulate = max(round(64 / (self.args.batch_size * self.args.world_size)), 1)
        self.params['weight_decay'] *= self.args.batch_size * self.args.world_size * self.accumulate / 64
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            util.weight_decay(self.model, self.params['weight_decay']),
            self.params['lr0'],
            self.params['momentum'],
            nesterov=True
        )
        
        # Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: (1 - x / self.args.epochs) * (1.0 - self.params['lrf']) + self.params['lrf']
        )
        
        # Warmup scheduler
        self.num_warmup = max(round(self.params['warmup_epochs'] * len(self.train_loader)), 100)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        if self.args.distributed:
            self.train_loader.sampler.set_epoch(epoch)
            
        # Disable mosaic for last 10 epochs
        if self.args.epochs - epoch == 10:
            self.train_loader.dataset.mosaic = False
        
        # Initialize progress bar
        pbar = enumerate(self.train_loader)
        if self.is_main_process:
            print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'box', 'cls'))
            pbar = tqdm.tqdm(pbar, total=len(self.train_loader))
        
        # Metrics
        avg_losses = {'box': util.AverageMeter(), 'cls': util.AverageMeter()}
        
        for i, (images, targets) in pbar:
            # Move data to device
            images = images.to(self.device, non_blocking=True).float() / 255.0
            
            # Warmup
            self.warmup(i + len(self.train_loader) * epoch)
            
            # Forward pass
            outputs = self.model(images)
            loss_box, loss_cls = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss = loss_box + loss_cls
            total_loss.backward()
            
            # Optimize
            if (i + len(self.train_loader) * epoch) % self.accumulate == 0:
                util.clip_gradients(self.model)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            avg_losses['box'].update(loss_box.item(), images.size(0))
            avg_losses['cls'].update(loss_cls.item(), images.size(0))
            
            # Update progress bar
            if self.is_main_process:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                pbar.set_description(('%10s' * 2 + '%10.3g' * 2) % (
                    f'{epoch + 1}/{self.args.epochs}',
                    memory,
                    avg_losses['box'].avg,
                    avg_losses['cls'].avg
                ))
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return avg_losses
    
    def warmup(self, step):
        """Learning rate warmup and momentum adjustment"""
        if step <= self.num_warmup:
            # Accumulate gradient iterations
            xp = [0, self.num_warmup]
            fp = [1, 64 / (self.args.batch_size * self.args.world_size)]
            self.accumulate = max(1, np.interp(step, xp, fp).round())
            
            # Adjust learning rate and momentum
            for j, param_group in enumerate(self.optimizer.param_groups):
                # Bias lr adjustment
                if j == 0:
                    fp = [self.params['warmup_bias_lr'], param_group['initial_lr'] * self.lr_scheduler.get_last_lr()[0]]
                else:
                    fp = [0.0, param_group['initial_lr'] * self.lr_scheduler.get_last_lr()[0]]
                
                param_group['lr'] = np.interp(step, xp, fp)
                
                # Momentum adjustment
                if 'momentum' in param_group:
                    fp = [self.params['warmup_momentum'], self.params['momentum']]
                    param_group['momentum'] = np.interp(step, xp, fp)
    
    @torch.no_grad()
    def validate(self, model=None):
        """Validate the model on validation set"""
        if model is None:
            model = self.model.module if self.args.distributed else self.model
        
        model.eval()
        device = next(model.parameters()).device
        
        # Initialize metrics
        stats = []
        iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
        
        pbar = tqdm.tqdm(self.val_loader, desc='Validating')
        for images, targets in pbar:
            images = images.to(device, non_blocking=True).float() / 255.0
            _, _, h, w = images.shape
            scale = torch.tensor((w, h, w, h), device=device)
            
            # Inference
            outputs = model(images)
            
            # NMS
            outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.nc)
            
            # Process each image in batch
            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                gt_cls = targets['cls'][idx].to(device)
                gt_box = targets['box'][idx].to(device)
                
                if output is None:
                    if gt_cls.shape[0]:
                        stats.append((torch.zeros(0, len(iou_thresholds), dtype=torch.bool, device=device),
                                    torch.zeros(0), torch.zeros(0), gt_cls.squeeze(-1)))
                    continue
                
                # Convert boxes to xyxy format
                pred_box = output[:, :4]
                pred_conf = output[:, 4]
                pred_cls = output[:, 5]
                
                # Evaluate
                if gt_cls.shape[0]:
                    gt_box = util.wh2xy(gt_box) * scale
                    tp, fp, _, _ = util.compute_ap(pred_box, gt_box, pred_cls, gt_cls, iou_thresholds)
                    stats.append((tp, fp, pred_conf, gt_cls.squeeze(-1)))
        
        # Compute metrics
        if len(stats):
            tp, fp, conf, gt_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
            results = ap_per_class(tp, fp, conf, gt_cls)
            precision, recall, map50, mean_ap = results[:4]
        else:
            precision, recall, map50, mean_ap = 0.0, 0.0, 0.0, 0.0
        
        if self.is_main_process:
            print('%10.3g' * 4 % (precision, recall, map50, mean_ap))
        
        model.train()
        return mean_ap, map50, recall, precision
    
    def save_model(self, model, metrics, epoch):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        # Convert to quantized model
        save_model = copy.deepcopy(model)
        save_model.eval()
        save_model.to('cpu')
        torch.ao.quantization.convert(save_model, inplace=True)
        save_model = torch.jit.script(save_model)
        
        # Save last and best models
        torch.jit.save(save_model, './weights/last.ts')
        if metrics[0] > self.best_ap:
            self.best_ap = metrics[0]
            torch.jit.save(save_model, './weights/best.ts')
        
        # Log metrics
        with open('weights/step.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'box', 'cls', 'Recall', 'Precision', 'mAP@50', 'mAP'])
            if epoch == 0:
                writer.writeheader()
            writer.writerow({
                'epoch': str(epoch + 1).zfill(3),
                'box': f'{metrics[0]:.3f}',
                'cls': f'{metrics[1]:.3f}',
                'Recall': f'{metrics[2]:.3f}',
                'Precision': f'{metrics[3]:.3f}',
                'mAP@50': f'{metrics[4]:.3f}',
                'mAP': f'{metrics[5]:.3f}'
            })
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.args.epochs):
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate and save model
            if self.is_main_process:
                val_metrics = self.validate()
                self.save_model(
                    self.model.module if self.args.distributed else self.model,
                    [train_metrics['box'].avg, train_metrics['cls'].avg, *val_metrics],
                    epoch
                )
        
        if self.is_main_process and self.writer:
            self.writer.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def setup_environment(args):
    """Setup distributed training environment"""
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    if args.local_rank == 0 and not os.path.exists('weights'):
        os.makedirs('weights')


def main():
    args = parse_args()
    setup_environment(args)
    
    with open('utils/args.yaml') as f:
        params = yaml.safe_load(f)
    
    if args.train:
        trainer = Trainer(args, params)
        trainer.train()
    
    if args.test:
        model = torch.jit.load('./weights/best.ts')
        metrics = Trainer(args, params).validate(model)
        print(f'mAP: {metrics[0]:.4f}, mAP50: {metrics[1]:.4f}')


if __name__ == "__main__":
    main()
