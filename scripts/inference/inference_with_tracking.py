#!/usr/bin/env python3
"""
Integrated YOLO Inference + Kalman Tracking Pipeline
Runs detection on videos and applies tracking to generate trajectories.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from loguru import logger
import sys
from typing import List, Dict, Tuple
import json
import csv
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from tracking.kalman_tracker import KalmanTracker
from tracking.trajectory import TrajectoryAnalyzer


class InferenceTracker:
    """Integrated detection and tracking pipeline."""
    
    def __init__(self, model_path: str, conf: float = 0.1, imgsz: int = 1280):
        """
        Initialize inference tracker.
        
        Args:
            model_path: Path to YOLO model weights
            conf: Confidence threshold
            imgsz: Input image size
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        
        # Initialize tracker
        tracker_config = {
            'max_disappeared': 15,  # Frames before removing track
            'max_distance': 100,    # Max distance for association
        }
        self.tracker = KalmanTracker(tracker_config)
        
        # Initialize trajectory analyzer
        traj_config = {
            'trajectory': {
                'analyze_bounce': True,
                'analyze_speed': True,
                'prediction_horizon': 5
            }
        }
        self.trajectory_analyzer = TrajectoryAnalyzer(traj_config)
        
        logger.info(f"Initialized with model: {model_path}")
        logger.info(f"Confidence: {conf}, Image size: {imgsz}")
    
    def process_video(self, video_path: str, output_dir: str = None) -> Dict:
        """
        Process video with detection and tracking.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for outputs (video + trajectory)
            
        Returns:
            Dictionary with tracking results and statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path("output/tracked_videos")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing: {video_path.name}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup output video
        output_video_path = output_dir / f"{video_path.stem}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Tracking data
        all_detections = []
        trajectory_data = []
        frame_count = 0
        detected_frames = 0
        
        # Per-frame CSV data
        csv_data = []
        
        # Trajectory visualization data (store positions per track)
        track_trajectories = {}
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                max_det=5
            )
            
            # Parse detections
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'centroid': (int(cx), int(cy)),
                        'confidence': conf,
                        'area': w * h
                    })
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store per-frame data for CSV
            frame_csv = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'detections': len(detections),
                'tracked_objects': len(tracked_objects)
            }
            
            # Add individual object data
            for i, obj in enumerate(tracked_objects):
                frame_csv[f'ball_{i+1}_x'] = obj['centroid'][0]
                frame_csv[f'ball_{i+1}_y'] = obj['centroid'][1]
                frame_csv[f'ball_{i+1}_confidence'] = obj.get('confidence', 0.0)
                frame_csv[f'ball_{i+1}_track_id'] = obj['id']
            
            csv_data.append(frame_csv)
            
            # Draw results
            frame_vis = frame.copy()
            
            for obj in tracked_objects:
                track_id = obj['id']
                bbox = obj['bbox']
                centroid = obj['centroid']
                conf = obj.get('confidence', 0.0)
                
                # Store trajectory point for this track
                if track_id not in track_trajectories:
                    track_trajectories[track_id] = []
                track_trajectories[track_id].append(centroid)
                
                # Draw bounding box
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label_y = y1 - 10
                else:
                    label_y = centroid[1] - 20
                
                # Draw center
                cv2.circle(frame_vis, centroid, 5, (0, 0, 255), -1)
                
                # Draw trajectory trail (last 30 points)
                if len(track_trajectories[track_id]) > 1:
                    points = track_trajectories[track_id][-30:]  # Last 30 frames
                    for i in range(1, len(points)):
                        # Fade effect: older points are more transparent
                        thickness = max(1, int(3 * (i / len(points))))
                        cv2.line(frame_vis, points[i-1], points[i], (255, 0, 255), thickness)
                
                # Draw track ID and confidence
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame_vis, label, (centroid[0] - 30, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Store trajectory point
                trajectory_data.append({
                    'frame': frame_count,
                    'track_id': track_id,
                    'centroid': centroid,
                    'bbox': bbox,
                    'confidence': conf
                })
            
            # Draw frame number
            cv2.putText(frame_vis, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw detection count
            if len(detections) > 0:
                detected_frames += 1
                cv2.putText(frame_vis, f"Detected: {len(tracked_objects)} ball(s)",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame_vis, "No detection",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame
            out.write(frame_vis)
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save per-frame CSV
        csv_path = output_dir / f"{video_path.stem}_detections.csv"
        if csv_data:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV saved: {csv_path}")
        
        # Analyze trajectory
        if trajectory_data:
            positions = [pt['centroid'] for pt in trajectory_data]
            
            # Smooth trajectory
            smoothed_positions = self.trajectory_analyzer.smooth_trajectory(positions)
            
            # Calculate speed
            speeds = self.trajectory_analyzer.calculate_speed(smoothed_positions, fps)
            
            # Detect bounces
            bounces = self.trajectory_analyzer.detect_bounces(smoothed_positions)
            
            # Save trajectory data
            trajectory_output = {
                'video': str(video_path.name),
                'total_frames': total_frames,
                'detected_frames': detected_frames,
                'detection_rate': detected_frames / total_frames,
                'trajectory_points': len(trajectory_data),
                'raw_trajectory': trajectory_data,
                'smoothed_positions': smoothed_positions,
                'speeds': speeds,
                'avg_speed': float(np.mean(speeds)) if speeds else 0,
                'max_speed': float(np.max(speeds)) if speeds else 0,
                'bounces': bounces,
                'bounce_count': len(bounces)
            }
            
            # Save to JSON
            trajectory_json_path = output_dir / f"{video_path.stem}_trajectory.json"
            with open(trajectory_json_path, 'w') as f:
                json.dump(trajectory_output, f, indent=2)
            
            logger.info(f"Trajectory saved: {trajectory_json_path}")
        
        # Results summary
        results_summary = {
            'video': str(video_path.name),
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'detection_rate': f"{detected_frames/total_frames*100:.1f}%",
            'output_video': str(output_video_path),
            'csv_file': str(csv_path) if csv_data else None,
            'trajectory_file': str(trajectory_json_path) if trajectory_data else None,
            'avg_speed': f"{np.mean(speeds):.1f} px/s" if speeds else "N/A",
            'bounces_detected': len(bounces) if trajectory_data else 0
        }
        
        logger.info("=" * 60)
        logger.info(f"✅ Processing complete: {video_path.name}")
        logger.info(f"Detection rate: {results_summary['detection_rate']}")
        logger.info(f"Output: {output_video_path}")
        logger.info("=" * 60)
        
        return results_summary


def main():
    """Main function to process all test videos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Detection + Kalman Tracking')
    parser.add_argument('--model', type=str,
                       default='runs/detect/local_finetune_optimized3/weights/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--video', type=str, default=None,
                       help='Single video to process (optional)')
    parser.add_argument('--input-dir', type=str,
                       default='data/raw/25_nov_2025',
                       help='Directory with test videos')
    parser.add_argument('--output-dir', type=str,
                       default='output/tracked_videos',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = InferenceTracker(args.model, args.conf, args.imgsz)
    
    # Process videos
    all_results = []
    
    if args.video:
        # Process single video
        results = tracker.process_video(args.video, args.output_dir)
        all_results.append(results)
    else:
        # Process all videos in directory
        input_dir = Path(args.input_dir)
        video_files = sorted(list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.mov')))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            try:
                results = tracker.process_video(str(video_path), args.output_dir)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {e}")
    
    # Save summary
    summary_path = Path(args.output_dir) / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ All videos processed. Summary: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'Video':<20} {'Frames':<8} {'Detected':<10} {'Rate':<10} {'Avg Speed':<15} {'Bounces':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['video']:<20} {r['total_frames']:<8} {r['detected_frames']:<10} "
              f"{r['detection_rate']:<10} {r['avg_speed']:<15} {r['bounces_detected']:<10}")
    print("=" * 80)


if __name__ == '__main__':
    main()
