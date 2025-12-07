"""
Trajectory analysis module.
Analyzes ball trajectory for speed, bounce detection, and prediction.
"""

import numpy as np
from scipy.signal import savgol_filter
from loguru import logger


class TrajectoryAnalyzer:
    """Analyze ball trajectory."""
    
    def __init__(self, config):
        """
        Initialize trajectory analyzer.
        
        Args:
            config: Dictionary with trajectory configuration
        """
        self.config = config
        self.analyze_bounce = config['trajectory'].get('analyze_bounce', True)
        self.analyze_speed = config['trajectory'].get('analyze_speed', True)
        self.prediction_horizon = config['trajectory'].get('prediction_horizon', 5)
    
    def smooth_trajectory(self, positions, window_length=5, polyorder=2):
        """
        Smooth trajectory using Savitzky-Golay filter.
        
        Args:
            positions: List of (x, y) positions
            window_length: Window length for smoothing
            polyorder: Polynomial order
            
        Returns:
            Smoothed positions
        """
        if len(positions) < window_length:
            return positions
        
        positions = np.array(positions)
        
        # Smooth x and y separately
        x_smooth = savgol_filter(positions[:, 0], window_length, polyorder)
        y_smooth = savgol_filter(positions[:, 1], window_length, polyorder)
        
        smoothed = np.column_stack([x_smooth, y_smooth])
        return smoothed.tolist()
    
    def calculate_speed(self, positions, fps=30):
        """
        Calculate instantaneous speed.
        
        Args:
            positions: List of (x, y) positions
            fps: Frames per second
            
        Returns:
            List of speeds in pixels/second
        """
        if len(positions) < 2:
            return []
        
        positions = np.array(positions)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        speeds = distances * fps
        
        return speeds.tolist()
    
    def detect_bounces(self, positions, threshold=20):
        """
        Detect bounces in trajectory (sudden changes in y-direction).
        
        Args:
            positions: List of (x, y) positions
            threshold: Minimum change in y to consider a bounce
            
        Returns:
            List of bounce frame indices
        """
        if len(positions) < 3:
            return []
        
        positions = np.array(positions)
        y_positions = positions[:, 1]
        
        # Calculate second derivative (acceleration in y)
        y_accel = np.diff(y_positions, n=2)
        
        # Find peaks (bounces)
        bounces = []
        for i in range(1, len(y_accel) - 1):
            if abs(y_accel[i]) > threshold:
                if (y_accel[i - 1] < y_accel[i] > y_accel[i + 1]):
                    bounces.append(i + 1)  # +1 to account for diff
        
        return bounces
    
    def predict_position(self, positions, n_steps=5):
        """
        Predict future positions using linear extrapolation.
        
        Args:
            positions: List of recent (x, y) positions
            n_steps: Number of steps to predict ahead
            
        Returns:
            List of predicted positions
        """
        if len(positions) < 2:
            return []
        
        positions = np.array(positions)
        
        # Fit linear model to recent positions
        n_recent = min(10, len(positions))
        recent_positions = positions[-n_recent:]
        
        # Calculate average velocity
        velocities = np.diff(recent_positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        
        # Predict future positions
        last_position = positions[-1]
        predictions = []
        
        for i in range(1, n_steps + 1):
            pred_pos = last_position + avg_velocity * i
            predictions.append(tuple(pred_pos))
        
        return predictions
    
    def calculate_trajectory_stats(self, positions, fps=30):
        """
        Calculate comprehensive trajectory statistics.
        
        Args:
            positions: List of (x, y) positions
            fps: Frames per second
            
        Returns:
            Dictionary of statistics
        """
        if len(positions) < 2:
            return {}
        
        positions = np.array(positions)
        
        # Calculate total distance
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Calculate speeds
        speeds = distances * fps
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        
        # Calculate trajectory bounds
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        stats = {
            'total_distance': float(total_distance),
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed),
            'bounds': {
                'min_x': float(min_x),
                'max_x': float(max_x),
                'min_y': float(min_y),
                'max_y': float(max_y)
            },
            'duration_frames': len(positions),
            'duration_seconds': len(positions) / fps
        }
        
        # Detect bounces
        if self.analyze_bounce:
            bounces = self.detect_bounces(positions)
            stats['num_bounces'] = len(bounces)
            stats['bounce_frames'] = bounces
        
        return stats
