"""
Advanced ML-based detector with multi-stage fallback system.
Primary: YOLO detection
Fallback 1: HSV color masking
Fallback 2: Motion-based foreground detection
Fallback 3: Hough circle detection
Fallback 4: Kalman prediction-based ROI search
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class YOLODetectorWithFallback:
    """
    YOLO-based ball detector with intelligent multi-stage fallback.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize YOLO detector with fallback mechanisms.
        
        Args:
            config: Configuration dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Ultralytics required. "
                "Install: pip install torch ultralytics"
            )
        
        self.config = config or {}
        
        # YOLO parameters
        model_path = self.config.get('model_path', 'yolov8n.pt')
        self.yolo_confidence = self.config.get('yolo_confidence', 0.4)
        self.yolo_iou = self.config.get('yolo_iou', 0.45)
        self.ball_class_ids = self.config.get('ball_class_ids', [32, 37])  # sports ball, baseball
        
        # YOLO bounding box size filters (to reject people/large objects)
        self.yolo_min_bbox_area = self.config.get('yolo_min_bbox_area', 50)
        self.yolo_max_bbox_area = self.config.get('yolo_max_bbox_area', 5000)
        self.yolo_max_bbox_width = self.config.get('yolo_max_bbox_width', 150)
        self.yolo_max_bbox_height = self.config.get('yolo_max_bbox_height', 150)
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Fallback parameters
        self.enable_fallbacks = self.config.get('enable_fallbacks', True)
        self.fallback_confidence = self.config.get('fallback_confidence', 0.3)
        
        # HSV color parameters (for white/red ball)
        self.ball_type = self.config.get('ball_type', 'white')
        if self.ball_type == 'white':
            self.lower_color = np.array(self.config.get('lower_white', [0, 0, 180]))
            self.upper_color = np.array(self.config.get('upper_white', [180, 40, 255]))
        else:
            self.lower_color1 = np.array(self.config.get('lower_red1', [0, 100, 100]))
            self.upper_color1 = np.array(self.config.get('upper_red1', [10, 255, 255]))
            self.lower_color2 = np.array(self.config.get('lower_red2', [160, 100, 100]))
            self.upper_color2 = np.array(self.config.get('upper_red2', [180, 255, 255]))
        
        # Size filters
        self.min_radius = self.config.get('min_radius', 3)
        self.max_radius = self.config.get('max_radius', 30)
        self.min_area = self.config.get('min_area', 20)
        self.max_area = self.config.get('max_area', 3000)
        self.min_circularity = self.config.get('min_circularity', 0.5)
        
        # Motion detection (background subtractor)
        self.use_motion_detection = self.config.get('use_motion_detection', True)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Hough circle detection parameters
        self.hough_dp = self.config.get('hough_dp', 1.2)
        self.hough_min_dist = self.config.get('hough_min_dist', 30)
        self.hough_param1 = self.config.get('hough_param1', 50)
        self.hough_param2 = self.config.get('hough_param2', 30)
        
        # Predicted position for ROI-based search
        self.last_detection = None
        self.roi_expansion = self.config.get('roi_expansion', 100)
        
        # Statistics
        self.detection_stats = {
            'yolo': 0,
            'color': 0,
            'motion': 0,
            'hough': 0,
            'prediction': 0,
            'failed': 0
        }
    
    def detect(self, frame: np.ndarray, predicted_pos: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Detect ball using YOLO with fallback mechanisms.
        
        Args:
            frame: Input BGR frame
            predicted_pos: Kalman-predicted position (x, y) for ROI search
            
        Returns:
            List of detections with 'source' field indicating detection method
        """
        # Stage 1: YOLO Detection
        detections = self._yolo_detect(frame)
        
        if detections and detections[0]['confidence'] >= self.yolo_confidence:
            self.detection_stats['yolo'] += 1
            self.last_detection = detections[0]
            return detections
        
        if not self.enable_fallbacks:
            self.detection_stats['failed'] += 1
            return []
        
        # Stage 2: ROI-based search if we have prediction
        if predicted_pos is not None:
            roi_detections = self._roi_based_search(frame, predicted_pos)
            if roi_detections:
                self.detection_stats['prediction'] += 1
                self.last_detection = roi_detections[0]
                return roi_detections
        
        # Stage 3: HSV Color-based fallback
        color_detections = self._color_detect(frame)
        if color_detections:
            self.detection_stats['color'] += 1
            self.last_detection = color_detections[0]
            return color_detections
        
        # Stage 4: Motion-based detection
        if self.use_motion_detection:
            motion_detections = self._motion_detect(frame)
            if motion_detections:
                self.detection_stats['motion'] += 1
                self.last_detection = motion_detections[0]
                return motion_detections
        
        # Stage 5: Hough circle detection (last resort)
        hough_detections = self._hough_detect(frame)
        if hough_detections:
            self.detection_stats['hough'] += 1
            self.last_detection = hough_detections[0]
            return hough_detections
        
        # All methods failed
        self.detection_stats['failed'] += 1
        return []
    
    def _yolo_detect(self, frame: np.ndarray) -> List[Dict]:
        """Primary YOLO detection with size filtering."""
        results = self.model(frame, conf=self.yolo_confidence, iou=self.yolo_iou, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Filter for ball classes only
                if class_id not in self.ball_class_ids:
                    continue
                
                confidence = float(box.conf[0])
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Apply size filters to reject people/large objects
                bbox_area = w * h
                if bbox_area < self.yolo_min_bbox_area or bbox_area > self.yolo_max_bbox_area:
                    continue
                if w > self.yolo_max_bbox_width or h > self.yolo_max_bbox_height:
                    continue
                
                # Calculate centroid
                cx, cy = x + w//2, y + h//2
                
                # Estimate radius
                radius = min(w, h) // 2
                area = w * h
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'radius': radius,
                    'confidence': confidence,
                    'area': area,
                    'source': 'yolo'
                })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:1]  # Return only best detection
    
    def _color_detect(self, frame: np.ndarray) -> List[Dict]:
        """Fallback: HSV color-based detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        if self.ball_type == 'white':
            mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        else:
            mask1 = cv2.inRange(hsv, self.lower_color1, self.upper_color1)
            mask2 = cv2.inRange(hsv, self.lower_color2, self.upper_color2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return self._extract_detections_from_mask(mask, frame.shape, 'color')
    
    def _motion_detect(self, frame: np.ndarray) -> List[Dict]:
        """Fallback: Motion-based foreground detection."""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return self._extract_detections_from_mask(fg_mask, frame.shape, 'motion')
    
    def _hough_detect(self, frame: np.ndarray) -> List[Dict]:
        """Fallback: Hough circle detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detections = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                area = np.pi * r * r
                
                if area < self.min_area or area > self.max_area:
                    continue
                
                detections.append({
                    'bbox': (x-r, y-r, 2*r, 2*r),
                    'centroid': (x, y),
                    'radius': r,
                    'confidence': self.fallback_confidence,
                    'area': area,
                    'source': 'hough'
                })
        
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections[:1]
    
    def _roi_based_search(self, frame: np.ndarray, predicted_pos: Tuple[int, int]) -> List[Dict]:
        """Search in ROI around predicted position."""
        px, py = predicted_pos
        h, w = frame.shape[:2]
        
        # Define ROI
        x1 = max(0, px - self.roi_expansion)
        y1 = max(0, py - self.roi_expansion)
        x2 = min(w, px + self.roi_expansion)
        y2 = min(h, py + self.roi_expansion)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        # Try color detection in ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        if self.ball_type == 'white':
            mask = cv2.inRange(hsv_roi, self.lower_color, self.upper_color)
        else:
            mask1 = cv2.inRange(hsv_roi, self.lower_color1, self.upper_color1)
            mask2 = cv2.inRange(hsv_roi, self.lower_color2, self.upper_color2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] > 0:
                # Convert ROI coordinates to full frame coordinates
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
            else:
                continue
            
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            
            detections.append({
                'bbox': (x_box + x1, y_box + y1, w_box, h_box),
                'centroid': (cx, cy),
                'radius': radius,
                'confidence': self.fallback_confidence,
                'area': area,
                'source': 'roi_prediction'
            })
        
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections[:1]
    
    def _extract_detections_from_mask(self, mask: np.ndarray, frame_shape: Tuple, source: str) -> List[Dict]:
        """Extract ball detections from binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = frame_shape[:2]
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < self.min_circularity:
                    continue
            else:
                continue
            
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)
            
            # Filter edge detections
            if cx < 50 or cx > w-50 or cy < 50 or cy > h-50:
                continue
            
            # Bounding box
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            
            detections.append({
                'bbox': (x_box, y_box, w_box, h_box),
                'centroid': (cx, cy),
                'radius': radius,
                'confidence': self.fallback_confidence,
                'area': area,
                'source': source
            })
        
        # Sort by circularity and area
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections[:1]
    
    def get_detection_stats(self) -> Dict:
        """Get detection method statistics."""
        total = sum(self.detection_stats.values())
        
        if total == 0:
            return self.detection_stats
        
        stats_with_percentage = {}
        for method, count in self.detection_stats.items():
            percentage = (count / total) * 100
            stats_with_percentage[method] = {
                'count': count,
                'percentage': f'{percentage:.1f}%'
            }
        
        return stats_with_percentage
