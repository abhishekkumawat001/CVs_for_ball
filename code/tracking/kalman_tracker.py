import numpy as np
import cv2
from typing import List, Dict
from scipy.spatial import distance


class KalmanTracker:
    """Kalman filter-based ball tracker."""
    
    def __init__(self, config: dict = None):
        """
        Initialize Kalman tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_disappeared = self.config.get('max_disappeared', 10)
        self.max_distance = self.config.get('max_distance', 50)
        
        self.tracks = {}
        self.next_track_id = 0
        self.disappeared = {}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from detector
            
        Returns:
            List of tracked objects with IDs
        """
        # Only track the single best detection (highest confidence/largest area)
        if len(detections) > 1:
            # Sort by confidence if available, otherwise by area
            if 'confidence' in detections[0]:
                detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
            else:
                detections = sorted(detections, key=lambda x: x.get('area', 0), reverse=True)
            detections = detections[:1]  # Keep only the best detection
        
        # If no tracks exist, create new ones
        if len(self.tracks) == 0:
            for detection in detections:
                self._create_track(detection)
                break  # Only create one track (for single ball)
        
        # If no detections, mark tracks as disappeared
        elif len(detections) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                
                # Remove lost tracks
                if self.disappeared[track_id] > self.max_disappeared:
                    self._delete_track(track_id)
        
        # Match detections to existing tracks
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
            
            detection_centroids = [d['centroid'] for d in detections]
            
            # Compute distance matrix
            D = distance.cdist(np.array(track_centroids), np.array(detection_centroids))
            
            # Hungarian algorithm for assignment
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update matched tracks
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                track_id = track_ids[row]
                self._update_track(track_id, detections[col])
                
                self.disappeared[track_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Mark unmatched tracks as disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                track_id = track_ids[row]
                self.disappeared[track_id] += 1
                
                if self.disappeared[track_id] > self.max_disappeared:
                    self._delete_track(track_id)
            
            # Create new tracks for unmatched detections (only if we have no active tracks)
            unused_cols = set(range(D.shape[1])) - used_cols
            if len(self.tracks) == 0:  # Only create new track if no existing tracks
                for col in unused_cols:
                    self._create_track(detections[col])
                    break  # Only track one ball
        
        # Return tracked objects
        tracked_objects = []
        for track_id, track in self.tracks.items():
            tracked_objects.append({
                'id': track_id,
                'centroid': track['centroid'],
                'bbox': track.get('bbox', None),
                'confidence': track.get('confidence', 1.0)
            })
        
        return tracked_objects
    
    def _create_track(self, detection: Dict):
        """Create new track from detection."""
        x, y = detection['centroid']  # Changed from 'center' to 'centroid'
        
        # Initialize Kalman filter
        kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        
        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial state
        kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        
        # Store track
        self.tracks[self.next_track_id] = {
            'kalman': kf,
            'centroid': (x, y),
            'bbox': detection.get('bbox'),
            'confidence': detection.get('confidence', 1.0)
        }
        
        self.disappeared[self.next_track_id] = 0
        self.next_track_id += 1
    
    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection."""
        track = self.tracks[track_id]
        kf = track['kalman']
        
        x, y = detection['centroid']  # Changed from 'center' to 'centroid'
        
        # Predict
        kf.predict()
        
        # Update
        measurement = np.array([[x], [y]], dtype=np.float32)
        kf.correct(measurement)
        
        # Update track info
        track['centroid'] = (int(kf.statePost[0]), int(kf.statePost[1]))
        track['bbox'] = detection.get('bbox')
        track['confidence'] = detection.get('confidence', 1.0)
    
    def _delete_track(self, track_id: int):
        """Delete lost track."""
        del self.tracks[track_id]
        del self.disappeared[track_id]