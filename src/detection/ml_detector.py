import cv2
import numpy as np
from typing import List, Dict

try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLBasedDetector:
    """ML-based ball detector using YOLO."""
    
    def __init__(self, config: dict = None):
        """
        Initialize ML detector.
        
        Args:
            config: Configuration with model_path
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Ultralytics are required for ML detection. "
                "Install with: pip install torch ultralytics"
            )
        
        self.config = config or {}
        model_path = self.config.get('model_path', 'yolov8n.pt')
        self.confidence_threshold = self.config.get('confidence', 0.5)
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Sports ball class ID in COCO dataset
        self.ball_class_id = 32  # sports ball
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect ball using YOLO.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter for sports ball class
                if cls == self.ball_class_id and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    x_box = int(x1)
                    y_box = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    radius = int(min(w, h) / 2)
                    
                    detections.append({
                        'bbox': (x_box, y_box, w, h),
                        'centroid': (cx, cy),
                        'radius': radius,
                        'confidence': conf,
                        'class': 'ball'
                    })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections