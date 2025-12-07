#!/usr/bin/env python3
"""
Standardize all CSV annotation files to simple format: frame,x,y,visible
Converts from detailed format (timestamp, confidence, etc.) to basic tracking format.
"""

import pandas as pd
import os
from pathlib import Path

def standardize_csv(input_file, output_file):
    """
    Convert CSV to standardized format: frame,x,y,visible
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
    """
    df = pd.read_csv(input_file)
    
    # Check if already in correct format
    if list(df.columns) == ['frame', 'x', 'y', 'visible']:
        print(f"✓ {os.path.basename(input_file)} - Already in correct format")
        return
    
    # Create standardized dataframe
    standardized = pd.DataFrame()
    
    if 'ball_1_x' in df.columns and 'ball_1_y' in df.columns:
        # Convert from detailed format
        standardized['frame'] = df['frame']
        
        # Extract x, y coordinates
        standardized['x'] = df['ball_1_x'].apply(lambda v: int(v) if pd.notna(v) else -1)
        standardized['y'] = df['ball_1_y'].apply(lambda v: int(v) if pd.notna(v) else -1)
        
        # Set visible flag: 1 if coordinates exist, 0 if missing
        standardized['visible'] = df.apply(
            lambda row: 1 if pd.notna(row['ball_1_x']) and pd.notna(row['ball_1_y']) else 0,
            axis=1
        )
    else:
        # Unknown format
        print(f"⚠ {os.path.basename(input_file)} - Unknown format, skipping")
        return
    
    # Save standardized CSV
    standardized.to_csv(output_file, index=False)
    print(f"✓ {os.path.basename(input_file)} - Converted to standard format ({len(standardized)} frames)")

def main():
    """Process all CSV files in annotations directory"""
    annotations_dir = Path(__file__).parent.parent / 'annotations'
    
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    print("=" * 60)
    print("Standardizing CSV Annotation Files")
    print("Format: frame,x,y,visible")
    print("=" * 60)
    print()
    
    csv_files = sorted(annotations_dir.glob('*_detections.csv'))
    
    if not csv_files:
        print("No CSV files found in annotations directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    for csv_file in csv_files:
        try:
            standardize_csv(csv_file, csv_file)
        except Exception as e:
            print(f"✗ {csv_file.name} - Error: {e}")
    
    print()
    print("=" * 60)
    print("✅ Standardization Complete")
    print("=" * 60)

if __name__ == '__main__':
    main()
