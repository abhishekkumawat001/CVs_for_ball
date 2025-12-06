import argparse
import cv2
import yaml
from pathlib import Path
from detection.color_detector import ColorBasedDetector
from detection.contour_detector import ContourBasedDetector
from detection.ml_detector import MLBasedDetector
from detection.yolo_detector import YOLODetectorWithFallback
from tracking.kalman_tracker import KalmanTracker
from tracking.optical_flow_tracker import OpticalFlowTracker
from postprocessing.visualizer import Visualizer
from postprocessing.video_writer import VideoWriter
from preprocessing.preprocessor import Preprocessor
from utils.logger import setup_logger
from utils.video_utils import VideoReader

logger = setup_logger(__name__)


class CricketBallTracker:
    """Main cricket ball tracking system."""
    
    def __init__(self, detection_method='color', tracking_method='kalman', 
                 detection_config=None, tracking_config=None, preprocessing_config=None,
                 show_preview=False, use_preprocessing=True):
        """
        Initialize tracker.
        
        Args:
            detection_method: 'color', 'contour', 'ml', or 'yolo'
            tracking_method: 'kalman' or 'optical_flow'
            detection_config: Path to detection config file
            tracking_config: Path to tracking config file
            preprocessing_config: Path to preprocessing config file
            show_preview: Show live tracking window
            use_preprocessing: Enable preprocessing pipeline
        """
        self.show_preview = show_preview
        self.use_preprocessing = use_preprocessing
        
        # Load configs
        self.detection_config = self._load_config(detection_config or 'config/detection_config.yaml')
        self.tracking_config = self._load_config(tracking_config or 'config/tracking_config.yaml')
        self.preprocessing_config = self._load_config(preprocessing_config or 'config/preprocessing_config.yaml')
        
        # Initialize preprocessor
        if self.use_preprocessing:
            self.preprocessor = Preprocessor(self.preprocessing_config.get('preprocessor', {}))
            logger.info("Preprocessing enabled")
        else:
            self.preprocessor = None
            logger.info("Preprocessing disabled")
        
        # Initialize detector
        if detection_method == 'color':
            self.detector = ColorBasedDetector(self.detection_config.get('color_detector', {}))
        elif detection_method == 'contour':
            self.detector = ContourBasedDetector(self.detection_config.get('contour_detector', {}))
        elif detection_method == 'ml':
            self.detector = MLBasedDetector(self.detection_config.get('ml_detector', {}))
        elif detection_method == 'yolo':
            self.detector = YOLODetectorWithFallback(self.detection_config.get('yolo_detector', {}))
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
        
        # Initialize tracker
        if tracking_method == 'kalman':
            self.tracker = KalmanTracker(self.tracking_config)
        elif tracking_method == 'optical_flow':
            self.tracker = OpticalFlowTracker(self.tracking_config)
        else:
            raise ValueError(f"Unknown tracking method: {tracking_method}")
        
        # Initialize visualizer with config (or empty dict if none)
        viz_config = self.tracking_config.get('visualization', {})
        self.visualizer = Visualizer(viz_config)
        
        logger.info(f"Initialized with {detection_method} detector and {tracking_method} tracker")
    
    def _load_config(self, config_path):
        """Load YAML config file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def process_video(self, input_path, output_path=None, save_trajectory=False):
        """
        Process video file and track cricket ball.
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            save_trajectory: Save trajectory data to CSV
        """
        logger.info(f"Processing video: {input_path}")
        
        # Open video
        reader = VideoReader(input_path)
        if not reader.is_opened():
            logger.error(f"Failed to open video: {input_path}")
            return
        
        # Setup output
        writer = None
        if output_path:
            writer = VideoWriter(
                output_path,
                fps=reader.fps,
                frame_size=(reader.width, reader.height)
            )
        
        trajectories = []
        frame_count = 0
        detection_count = 0
        
        logger.info(f"Video: {reader.width}x{reader.height} @ {reader.fps} fps, {reader.frame_count} frames")
        
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                
                # Preprocessing
                if self.preprocessor:
                    processed_frame = self.preprocessor.process(frame)
                else:
                    processed_frame = frame
                
                # Get Kalman prediction for ROI-based search (if available)
                predicted_pos = None
                if hasattr(self.tracker, 'get_predicted_position'):
                    predicted_pos = self.tracker.get_predicted_position()
                
                # Detect ball (pass prediction if YOLO detector)
                if hasattr(self.detector, 'detect') and 'predicted_pos' in self.detector.detect.__code__.co_varnames:
                    detections = self.detector.detect(processed_frame, predicted_pos=predicted_pos)
                else:
                    detections = self.detector.detect(processed_frame)
                
                # Update tracker
                tracked_objects = self.tracker.update(detections)
                
                # Visualize on original frame
                annotated_frame = self.visualizer.draw_detections(frame.copy(), tracked_objects)
                
                # Save trajectory
                if tracked_objects:
                    detection_count += 1
                    for obj in tracked_objects:
                        trajectories.append({
                            'frame': frame_count,
                            'x': obj['centroid'][0],
                            'y': obj['centroid'][1],
                            'timestamp': frame_count / reader.fps
                        })
                
                # Write output
                if writer:
                    writer.write(annotated_frame)
                
                # Show preview
                if self.show_preview:
                    cv2.imshow('Cricket Ball Tracker', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User interrupted")
                        break
                    elif key == ord('p'):
                        # Pause
                        logger.info("Paused - press any key to continue")
                        cv2.waitKey(0)
                
                frame_count += 1
                
                # Progress
                if frame_count % 30 == 0:
                    progress = (frame_count / reader.frame_count) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{reader.frame_count})")
        
        finally:
            reader.release()
            if writer:
                writer.release()
            if self.show_preview:
                cv2.destroyAllWindows()
        
        # Save trajectory
        if save_trajectory and trajectories:
            import pandas as pd
            df = pd.DataFrame(trajectories)
            trajectory_path = output_path.replace('.mp4', '_trajectory.csv') if output_path else 'trajectory.csv'
            Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(trajectory_path, index=False)
            logger.info(f"Trajectory saved: {trajectory_path}")
        
        # Summary
        detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
        logger.info(f"Processing complete!")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Ball detected in: {detection_count}/{frame_count} frames ({detection_rate:.1f}%)")
        if output_path:
            logger.info(f"Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cricket Ball Tracking System')
    
    # Changed from --video to --input
    parser.add_argument('--input', '--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video file (default: output/videos/tracked.mp4)')
    parser.add_argument('--detector', '--detection-method', type=str, 
                        choices=['color', 'contour', 'ml', 'yolo'], default='yolo',
                        help='Detection method (default: yolo)')
    parser.add_argument('--tracker', '--tracking-method', type=str,
                        choices=['kalman', 'optical_flow'], default='kalman',
                        help='Tracking method (default: kalman)')
    parser.add_argument('--detection-config', type=str, default=None,
                        help='Path to detection config file')
    parser.add_argument('--tracking-config', type=str, default='config/tracking_config.yaml',
                        help='Path to tracking config file')
    parser.add_argument('--preprocessing-config', type=str, default='config/preprocessing_config.yaml',
                        help='Path to preprocessing config file')
    parser.add_argument('--no-preprocessing', action='store_true',
                        help='Disable preprocessing pipeline')
    parser.add_argument('--show', action='store_true',
                        help='Show live tracking preview')
    parser.add_argument('--save-trajectory', action='store_true',
                        help='Save trajectory data to CSV')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show detection method statistics')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to main config file (overrides other configs)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        Path('output/videos').mkdir(parents=True, exist_ok=True)
        args.output = 'output/videos/tracked.mp4'
    
    # Auto-select detection config based on detector type
    if args.detection_config is None:
        if args.detector == 'yolo':
            args.detection_config = 'config/yolo_config.yaml'
        else:
            args.detection_config = 'config/detection_config.yaml'
    
    # Initialize tracker
    tracker = CricketBallTracker(
        detection_method=args.detector,
        tracking_method=args.tracker,
        detection_config=args.detection_config,
        tracking_config=args.tracking_config,
        preprocessing_config=args.preprocessing_config,
        use_preprocessing=not args.no_preprocessing,
        show_preview=args.show
    )
    
    # Process video
    tracker.process_video(
        input_path=args.input,
        output_path=args.output,
        save_trajectory=args.save_trajectory
    )
    
    # Show detection statistics if requested
    if args.show_stats and hasattr(tracker.detector, 'get_detection_stats'):
        stats = tracker.detector.get_detection_stats()
        logger.info("\n" + "="*50)
        logger.info("DETECTION METHOD STATISTICS:")
        logger.info("="*50)
        for method, data in stats.items():
            if isinstance(data, dict):
                logger.info(f"  {method}: {data['count']} ({data['percentage']})")
            else:
                logger.info(f"  {method}: {data}")
        logger.info("="*50)


if __name__ == '__main__':
    main()