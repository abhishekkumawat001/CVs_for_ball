import cv2
import numpy as np
from typing import Tuple, Optional


class VideoReader:
    """Video reader wrapper for OpenCV."""
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file or camera index
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def is_opened(self) -> bool:
        """Check if video is opened."""
        return self.cap.isOpened()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame)
        """
        return self.cap.read()
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get frame at specific position.
        
        Args:
            frame_number: Frame index
            
        Returns:
            Frame or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


class VideoWriter:
    """Video writer wrapper for OpenCV."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 fourcc: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video
            fps: Frames per second
            frame_size: (width, height)
            fourcc: Video codec fourcc code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc_code,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        if self.writer:
            self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Save every nth frame
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count