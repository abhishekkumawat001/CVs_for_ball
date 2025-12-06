import cv2
import numpy as np
from typing import List, Dict


class ContourBasedDetector:
    """Detect cricket ball using contour and shape analysis."""
    
    def __init__(self, config: dict = None):
        """
        Initialize contour-based detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_radius = self.config.get('min_radius', 5)
        self.max_radius = self.config.get('max_radius', 50)
        self.min_circularity = self.config.get('min_circularity', 0.7)
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect ball using contour detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of detections with bbox, centroid, confidence
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Dilate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            # Fit circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                continue
            
            if circularity < self.min_circularity:
                continue
            
            # Bounding box
            x_box, y_box, w, h = cv2.boundingRect(contour)
            
            # Centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(x), int(y)
            
            detections.append({
                'bbox': (x_box, y_box, w, h),
                'centroid': (cx, cy),
                'radius': int(radius),
                'confidence': circularity,
                'area': area
            })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections