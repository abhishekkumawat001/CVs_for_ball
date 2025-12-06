"""
Optical flow-based tracker (alternative to Kalman filter).
"""

import cv2
import numpy as np
from loguru import logger


class OpticalFlowTracker:
    """Track cricket ball using optical flow."""
    
    def __init__(self, config):
        """
        Initialize optical flow tracker.
        
        Args:
            config: Dictionary with optical flow configuration
        """
        self.config = config
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=tuple(config['optical_flow'].get('win_size', [15, 15])),
            maxLevel=config['optical_flow'].get('max_level', 2),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     config['optical_flow'].get('criteria_count', 10),
                     config['optical_flow'].get('criteria_eps', 0.03))
        )
        
        self.tracks = {}
        self.next_track_id = 0
        self.prev_gray = None
        self.max_disappeared = config['tracking'].get('max_disappeared_frames', 10)
    
    def update(self, frame, detections):
        """
        Update tracks using optical flow.
        
        Args:
            frame: Current frame (BGR)
            detections: List of detections
            
        Returns:
            List of active tracks
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            # First frame, initialize tracks
            self.prev_gray = gray
            for detection in detections:
                self._create_track(detection)
            return self._get_active_tracks()
        
        # Track existing points using optical flow
        if len(self.tracks) > 0:
            old_points = np.array([track['points'][-1] for track in self.tracks.values()], 
                                 dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, old_points, None, **self.lk_params
            )
            
            # Update tracks
            for i, (track_id, track) in enumerate(list(self.tracks.items())):
                if status[i]:
                    # Update point
                    new_point = new_points[i].ravel()
                    track['points'].append(tuple(new_point))
                    track['age'] += 1
                else:
                    # Lost track
                    track['disappeared'] += 1
                    if track['disappeared'] > self.max_disappeared:
                        del self.tracks[track_id]
        
        # Add new tracks from detections
        for detection in detections:
            # Check if detection is close to existing track
            is_new = True
            for track in self.tracks.values():
                if len(track['points']) > 0:
                    last_point = track['points'][-1]
                    dist = np.linalg.norm(
                        np.array(detection['center']) - np.array(last_point)
                    )
                    if dist < 50:  # Threshold for matching
                        is_new = False
                        break
            
            if is_new:
                self._create_track(detection)
        
        self.prev_gray = gray
        return self._get_active_tracks()
    
    def _create_track(self, detection):
        """Create new track from detection."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = {
            'points': [detection['center']],
            'age': 0,
            'disappeared': 0
        }
    
    def _get_active_tracks(self):
        """Get list of active tracks."""
        active_tracks = []
        
        for track_id, track in self.tracks.items():
            if track['age'] < 5:  # Minimum track length
                continue
            
            active_tracks.append({
                'id': track_id,
                'position': track['points'][-1],
                'trajectory': track['points'][-30:],
                'age': track['age']
            })
        
        return active_tracks
