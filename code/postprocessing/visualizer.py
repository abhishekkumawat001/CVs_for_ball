import cv2
import numpy as np
from typing import List, Dict, Tuple


class Visualizer:
    """Visualization utilities for ball tracking."""
    
    def __init__(self, config: dict = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Colors
        self.bbox_color = tuple(self.config.get('bbox_color', [0, 255, 0]))  # Green
        self.trajectory_color = tuple(self.config.get('trajectory_color', [255, 0, 0]))  # Blue
        self.text_color = tuple(self.config.get('text_color', [255, 255, 255]))  # White
        
        # Line thickness
        self.bbox_thickness = self.config.get('bbox_thickness', 2)
        self.trajectory_thickness = self.config.get('trajectory_thickness', 2)
        
        # Font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.config.get('font_scale', 0.6)
        self.font_thickness = self.config.get('font_thickness', 2)
        
        # Trajectory history
        self.trajectory_history = {}
        self.max_trajectory_length = self.config.get('max_trajectory_length', 50)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections with bbox, centroid, id
            
        Returns:
            Annotated frame
        """
        for detection in detections:
            # Get detection info
            centroid = detection.get('centroid')
            bbox = detection.get('bbox')
            track_id = detection.get('id', -1)
            confidence = detection.get('confidence', 0.0)
            
            if centroid is None:
                continue
            
            # Draw bounding box
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(
                    frame,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    self.bbox_color,
                    self.bbox_thickness
                )
            
            # Draw center point
            cv2.circle(
                frame,
                (int(centroid[0]), int(centroid[1])),
                5,
                self.bbox_color,
                -1
            )
            
            # Draw ID and confidence
            if track_id >= 0:
                label = f"ID: {track_id}"
                if confidence > 0:
                    label += f" ({confidence:.2f})"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label,
                    self.font,
                    self.font_scale,
                    self.font_thickness
                )
                
                text_x = int(centroid[0]) - text_w // 2
                text_y = int(centroid[1]) - 15
                
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_h - 5),
                    (text_x + text_w + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    self.font,
                    self.font_scale,
                    self.text_color,
                    self.font_thickness
                )
            
            # Update trajectory history
            if track_id >= 0:
                if track_id not in self.trajectory_history:
                    self.trajectory_history[track_id] = []
                
                self.trajectory_history[track_id].append(centroid)
                
                # Limit trajectory length
                if len(self.trajectory_history[track_id]) > self.max_trajectory_length:
                    self.trajectory_history[track_id].pop(0)
                
                # Draw trajectory
                self._draw_trajectory(frame, track_id)
        
        return frame
    
    def _draw_trajectory(self, frame: np.ndarray, track_id: int):
        """Draw trajectory line for a track."""
        if track_id not in self.trajectory_history:
            return
        
        points = self.trajectory_history[track_id]
        
        if len(points) < 2:
            return
        
        # Draw line segments
        for i in range(1, len(points)):
            # Fade trajectory (older points are more transparent)
            alpha = i / len(points)
            thickness = max(1, int(self.trajectory_thickness * alpha))
            
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            cv2.line(
                frame,
                pt1,
                pt2,
                self.trajectory_color,
                thickness
            )
    
    def draw_trajectory_overlay(self, frame: np.ndarray, trajectory: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw full trajectory overlay.
        
        Args:
            frame: Input frame
            trajectory: List of (x, y) points
            
        Returns:
            Annotated frame
        """
        if len(trajectory) < 2:
            return frame
        
        # Convert to numpy array
        points = np.array(trajectory, dtype=np.int32)
        
        # Draw polyline
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=self.trajectory_color,
            thickness=self.trajectory_thickness
        )
        
        # Draw points
        for point in points:
            cv2.circle(
                frame,
                tuple(point),
                3,
                self.trajectory_color,
                -1
            )
        
        return frame
    
    def draw_info_panel(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """
        Draw information panel on frame.
        
        Args:
            frame: Input frame
            info: Dictionary with information to display
            
        Returns:
            Annotated frame
        """
        y_offset = 30
        line_height = 25
        
        for key, value in info.items():
            text = f"{key}: {value}"
            
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
            
            y_offset += line_height
        
        return frame
    
    def reset_trajectory(self, track_id: int = None):
        """
        Reset trajectory history.
        
        Args:
            track_id: Specific track ID to reset, or None to reset all
        """
        if track_id is None:
            self.trajectory_history.clear()
        elif track_id in self.trajectory_history:
            del self.trajectory_history[track_id]