"""
YOLOv8 Inference Script

Clean, modular inference implementation with:
- Image and video support
- Webcam streaming
- Batch processing
- Result visualization and export
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

from model import YOLO
from utils import non_max_suppression, scale, wh2xy


# =============================================================================
# Configuration
# =============================================================================

class InferenceConfig:
    """Inference configuration."""
    
    def __init__(self, config_path: str = 'config/config.yml'):
        self.class_names = self._load_class_names(config_path)
        self.colors = self._generate_colors(len(self.class_names))
    
    def _load_class_names(self, config_path: str) -> Dict[int, str]:
        """Load class names from config."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('names', {i: f'class_{i}' for i in range(80)})
        except Exception:
            return {i: f'class_{i}' for i in range(80)}
    
    def _generate_colors(self, num_classes: int) -> np.ndarray:
        """Generate unique colors for each class."""
        np.random.seed(42)
        return np.random.randint(0, 255, size=(num_classes, 3))
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name by ID."""
        return self.class_names.get(class_id, f'class_{class_id}')
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID."""
        color = self.colors[class_id % len(self.colors)]
        return tuple(map(int, color))


# =============================================================================
# Model Loading
# =============================================================================

class ModelLoader:
    """Handles model loading and initialization."""
    
    @staticmethod
    def load(weights_path: str, device: str = 'auto') -> Tuple[torch.nn.Module, torch.device]:
        """Load model from checkpoint."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        # Add YOLO to safe globals for loading
        torch.serialization.add_safe_globals([YOLO])
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        # Extract model
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
        
        # Prepare model
        model = model.float().fuse().eval()
        model.to(device)
        
        return model, device


# =============================================================================
# Image Preprocessing
# =============================================================================

class Preprocessor:
    """Handles image preprocessing for inference."""
    
    def __init__(self, input_size: int = 640, pad_color: Tuple[int, int, int] = (114, 114, 114)):
        self.input_size = input_size
        self.pad_color = pad_color
    
    def __call__(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple]:
        """
        Preprocess image for inference.
        
        Args:
            image: BGR image (H, W, C)
        
        Returns:
            Tuple of (tensor, original_shape, ratio_pad)
        """
        original_shape = image.shape[:2]  # (H, W)
        
        # Calculate resize ratio
        ratio = min(self.input_size / original_shape[0], 
                   self.input_size / original_shape[1])
        new_size = (int(round(original_shape[1] * ratio)), 
                   int(round(original_shape[0] * ratio)))
        
        # Resize
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        dw = (self.input_size - new_size[0]) / 2
        dh = (self.input_size - new_size[1]) / 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.pad_color
        )
        
        # Convert to tensor: BGR -> RGB, HWC -> CHW, normalize
        tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(tensor).unsqueeze(0)
        
        ratio_pad = ((ratio, ratio), (dw, dh))
        
        return tensor, original_shape, ratio_pad


# =============================================================================
# Post-processing
# =============================================================================

class Postprocessor:
    """Handles detection post-processing."""
    
    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def __call__(self, predictions: torch.Tensor, 
                 original_shape: Tuple[int, int],
                 ratio_pad: Tuple) -> List[List[float]]:
        """
        Post-process predictions.
        
        Args:
            predictions: Model output tensor
            original_shape: Original image shape (H, W)
            ratio_pad: Resize ratio and padding ((ratio_w, ratio_h), (pad_w, pad_h))
        
        Returns:
            List of detections [[x1, y1, x2, y2, conf, class_id], ...]
        """
        # Apply NMS
        detections_list = non_max_suppression(
            predictions,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        
        if not detections_list or len(detections_list[0]) == 0:
            return []
        
        detections = detections_list[0]
        
        # Scale coordinates to original image
        detections[:, :4] = scale(
            detections[:, :4],
            original_shape,
            ratio_pad[0],  # (ratio_w, ratio_h)
            ratio_pad[1]   # (pad_w, pad_h)
        )
        
        return detections.cpu().numpy().tolist()


# =============================================================================
# Visualization
# =============================================================================

class Visualizer:
    """Handles detection visualization."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.box_thickness = 2
    
    def draw(self, image: np.ndarray, detections: List[List[float]], 
             show_labels: bool = True, show_conf: bool = True) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input BGR image
            detections: List of [x1, y1, x2, y2, conf, class_id]
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
            
            color = self.config.get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Draw label
            if show_labels or show_conf:
                label = self.config.get_class_name(class_id)
                if show_conf:
                    label = f'{label} {conf:.2f}'
                
                # Get text size
                (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
                
                # Draw label background
                cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                
                # Draw text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        return annotated
    
    def draw_fps(self, image: np.ndarray, fps: float, num_detections: int) -> np.ndarray:
        """Draw FPS and detection count on image."""
        text = f'FPS: {fps:.1f} | Objects: {num_detections}'
        cv2.putText(image, text, (10, 30), self.font, 1, (0, 255, 0), 2)
        return image


# =============================================================================
# Inference Engine
# =============================================================================

class InferenceEngine:
    """Main inference engine."""
    
    def __init__(self, weights: str, config_path: str = 'config/config.yml',
                 input_size: int = 640, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'auto'):
        
        self.config = InferenceConfig(config_path)
        self.model, self.device = ModelLoader.load(weights, device)
        self.preprocessor = Preprocessor(input_size)
        self.postprocessor = Postprocessor(conf_threshold, iou_threshold)
        self.visualizer = Visualizer(self.config)
        
        print(f"Model loaded on {self.device}")
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[List[List[float]], float]:
        """
        Run inference on single image.
        
        Args:
            image: BGR image
        
        Returns:
            Tuple of (detections, inference_time)
        """
        # Preprocess
        tensor, original_shape, ratio_pad = self.preprocessor(image)
        tensor = tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        predictions = self.model(tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        detections = self.postprocessor(predictions, original_shape, ratio_pad)
        
        return detections, inference_time
    
    def process_image(self, image_path: str, output_path: Optional[str] = None,
                     show: bool = False) -> Tuple[np.ndarray, List]:
        """Process single image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        detections, inference_time = self.predict(image)
        annotated = self.visualizer.draw(image, detections)
        
        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f"Processed {image_path}: {len(detections)} detections, {fps:.1f} FPS")
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved to {output_path}")
        
        if show:
            cv2.imshow('Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated, detections
    
    def process_video(self, source: Union[str, int], output_path: Optional[str] = None,
                     show: bool = True, max_frames: Optional[int] = None):
        """Process video file or webcam stream."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Run inference
                detections, inference_time = self.predict(frame)
                total_time += inference_time
                frame_count += 1
                
                # Visualize
                annotated = self.visualizer.draw(frame, detections)
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                annotated = self.visualizer.draw_fps(annotated, current_fps, len(detections))
                
                # Write output
                if writer:
                    writer.write(annotated)
                
                # Display
                if show:
                    cv2.imshow('YOLOv8 Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Processed {frame_count} frames, Average FPS: {avg_fps:.1f}")
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Processing {len(image_files)} images...")
        
        for image_file in image_files:
            output_file = output_path / image_file.name
            try:
                self.process_image(str(image_file), str(output_file))
            except Exception as e:
                print(f"Error processing {image_file}: {e}")


# =============================================================================
# Detection Results
# =============================================================================

class DetectionResult:
    """Container for detection results."""
    
    def __init__(self, detections: List[List[float]], class_names: Dict[int, str]):
        self.detections = detections
        self.class_names = class_names
    
    @property
    def boxes(self) -> np.ndarray:
        """Get bounding boxes [x1, y1, x2, y2]."""
        if not self.detections:
            return np.array([])
        return np.array([d[:4] for d in self.detections])
    
    @property
    def confidences(self) -> np.ndarray:
        """Get confidence scores."""
        if not self.detections:
            return np.array([])
        return np.array([d[4] for d in self.detections])
    
    @property
    def class_ids(self) -> np.ndarray:
        """Get class IDs."""
        if not self.detections:
            return np.array([])
        return np.array([int(d[5]) for d in self.detections])
    
    @property
    def labels(self) -> List[str]:
        """Get class labels."""
        return [self.class_names.get(int(d[5]), 'unknown') for d in self.detections]
    
    def filter_by_class(self, class_ids: List[int]) -> 'DetectionResult':
        """Filter detections by class IDs."""
        filtered = [d for d in self.detections if int(d[5]) in class_ids]
        return DetectionResult(filtered, self.class_names)
    
    def filter_by_confidence(self, min_conf: float) -> 'DetectionResult':
        """Filter detections by minimum confidence."""
        filtered = [d for d in self.detections if d[4] >= min_conf]
        return DetectionResult(filtered, self.class_names)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'detections': self.detections,
            'num_objects': len(self.detections),
            'classes': list(set(self.labels))
        }
    
    def __len__(self) -> int:
        return len(self.detections)
    
    def __repr__(self) -> str:
        return f"DetectionResult({len(self)} detections)"


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Inference')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                       help='Model weights path')
    parser.add_argument('--source', type=str, default='0',
                       help='Image/video path, directory, or camera ID')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yml',
                       help='Config file path')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    parser.add_argument('--save', action='store_true',
                       help='Save results')
    parser.add_argument('--classes', nargs='+', type=int,
                       help='Filter by class IDs')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    engine = InferenceEngine(
        weights=args.weights,
        config_path=args.config,
        input_size=args.img_size,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        device=args.device
    )
    
    source = args.source
    
    # Determine source type
    if source.isdigit():
        # Webcam
        engine.process_video(
            int(source),
            output_path=str(output_dir / 'webcam_output.mp4') if args.save else None,
            show=True
        )
    elif os.path.isdir(source):
        # Directory
        engine.process_directory(source, str(output_dir))
    elif os.path.isfile(source):
        # File (image or video)
        ext = Path(source).suffix.lower()
        if ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
            engine.process_video(
                source,
                output_path=str(output_dir / f'output{ext}') if args.save else None,
                show=args.show
            )
        else:
            output_file = str(output_dir / Path(source).name) if args.save else None
            engine.process_image(source, output_file, show=args.show)
    else:
        print(f"Invalid source: {source}")
        return


if __name__ == '__main__':
    main()

