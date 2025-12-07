import cv2
import numpy as np
from typing import List, Dict


class ColorBasedDetector:
    """Detect cricket ball using color-based HSV filtering."""
    
    def __init__(self, config: dict = None):
        """
        Initialize color-based detector.
        
        Args:
            config: Configuration dictionary with color ranges
        """
        self.config = config or {}
        
        # Check if detecting white or red ball
        self.ball_type = self.config.get('ball_type', 'red')  # 'red' or 'white'
        
        if self.ball_type == 'white':
            # WHITE BALL detection (for white ball cricket)
            self.lower_white = np.array(self.config.get('lower_white', [0, 0, 180]))
            self.upper_white = np.array(self.config.get('upper_white', [180, 40, 255]))
        else:
            # RED BALL detection (default)
            self.lower_red1 = np.array(self.config.get('lower_red1', [0, 100, 100]))
            self.upper_red1 = np.array(self.config.get('upper_red1', [10, 255, 255]))
            self.lower_red2 = np.array(self.config.get('lower_red2', [160, 100, 100]))
            self.upper_red2 = np.array(self.config.get('upper_red2', [180, 255, 255]))
        
        self.min_radius = self.config.get('min_radius', 3)
        self.max_radius = self.config.get('max_radius', 30)
        self.min_area = self.config.get('min_area', 20)
        self.max_area = self.config.get('max_area', 3000)
        self.min_circularity = self.config.get('min_circularity', 0.5)
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect ball in frame using color filtering.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask based on ball type
        if self.ball_type == 'white':
            mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        else:
            # Red color (two ranges)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        h, w = frame.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # Filter by radius
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Calculate bounding box
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)
            
            # Skip if too close to edges (likely noise/clothing)
            if cx < 50 or cx > w-50 or cy < 50 or cy > h-50:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < self.min_circularity:
                    continue
                confidence = circularity
            else:
                continue
            
            detections.append({
                'bbox': (x_box, y_box, w_box, h_box),
                'centroid': (cx, cy),
                'radius': int(radius),
                'confidence': confidence,
                'area': area
            })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top 5 candidates
        return detections[:5]