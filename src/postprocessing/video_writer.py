"""
Video writer for saving processed output videos.
"""

import cv2
from pathlib import Path
from loguru import logger


class VideoWriter:
    """Write processed video to file."""
    
    def __init__(self, output_path, fps, frame_size, codec='mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height) tuple
            codec: Video codec fourcc code
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, fps, frame_size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Failed to open video writer: {output_path}")
        
        self.frame_count = 0
        logger.info(f"Video writer initialized: {output_path}")
    
    def write(self, frame):
        """
        Write frame to video.
        
        Args:
            frame: Frame to write
        """
        # Resize if needed
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            logger.info(f"Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def __del__(self):
        """Cleanup."""
        self.release()
