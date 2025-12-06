"""
Performance metrics for tracking evaluation.
"""

import numpy as np
from scipy.spatial.distance import euclidean


class TrackingMetrics:
    """Calculate tracking performance metrics."""
    
    @staticmethod
    def calculate_iou(bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x, y, w, h]
            bbox2: [x, y, w, h]
            
        Returns:
            IoU score (0-1)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        iou = inter_area / union_area
        return iou
    
    @staticmethod
    def calculate_tracking_accuracy(predicted_positions, ground_truth_positions, threshold=10):
        """
        Calculate tracking accuracy.
        
        Args:
            predicted_positions: List of predicted (x, y) positions
            ground_truth_positions: List of ground truth (x, y) positions
            threshold: Distance threshold in pixels
            
        Returns:
            Accuracy score (0-1)
        """
        if len(predicted_positions) != len(ground_truth_positions):
            return 0
        
        correct = 0
        for pred, gt in zip(predicted_positions, ground_truth_positions):
            if euclidean(pred, gt) <= threshold:
                correct += 1
        
        accuracy = correct / len(predicted_positions)
        return accuracy
    
    @staticmethod
    def calculate_precision_recall(detections, ground_truths, iou_threshold=0.5):
        """
        Calculate precision and recall for detections.
        
        Args:
            detections: List of detected bounding boxes
            ground_truths: List of ground truth bounding boxes
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall)
        """
        if len(detections) == 0:
            return 0, 0
        
        if len(ground_truths) == 0:
            return 0, 0
        
        # Match detections to ground truths
        matched_gt = set()
        true_positives = 0
        
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(ground_truths):
                if idx in matched_gt:
                    continue
                
                iou = TrackingMetrics.calculate_iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        precision = true_positives / len(detections)
        recall = true_positives / len(ground_truths)
        
        return precision, recall
    
    @staticmethod
    def calculate_mota(true_positives, false_positives, false_negatives, num_ground_truths):
        """
        Calculate Multiple Object Tracking Accuracy (MOTA).
        
        Args:
            true_positives: Number of true positives
            false_positives: Number of false positives
            false_negatives: Number of false negatives (misses)
            num_ground_truths: Total number of ground truth objects
            
        Returns:
            MOTA score
        """
        if num_ground_truths == 0:
            return 0
        
        mota = 1 - (false_negatives + false_positives) / num_ground_truths
        return mota
