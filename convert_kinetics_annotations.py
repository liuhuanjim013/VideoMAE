#!/usr/bin/env python3
"""
Convert Kinetics-400 CSV annotations to VideoMAE format.
VideoMAE expects: video_path,label_index
Kinetics provides: label,youtube_id,time_start,time_end,split,is_cc

Updated to handle training subset files with flexible filename patterns.
"""

import pandas as pd
import os
import argparse
import glob
from pathlib import Path

def detect_filename_patterns(input_dir):
    """
    Detect different filename patterns in the input directory.
    
    Returns:
        dict: Mapping of split names to CSV file paths
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    patterns = {}
    
    print(f"Found CSV files in {input_dir}:")
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        print(f"  {filename}")
        
        # Pattern 1: Standard kinetics format (train.csv, val.csv, test.csv)
        if filename in ['train.csv', 'val.csv', 'test.csv']:
            split_name = filename.replace('.csv', '')
            patterns[split_name] = csv_file
        
        # Pattern 2: Training subset format (top_10000_siglip_score_train.csv, etc.)
        elif 'train' in filename.lower() and not 'indices' in filename and not 'scores' in filename:
            patterns['train'] = csv_file
        elif 'eval' in filename.lower() and not 'indices' in filename and not 'scores' in filename:
            patterns['val'] = csv_file  # Map eval to val
        elif 'test' in filename.lower() and not 'indices' in filename and not 'scores' in filename:
            patterns['test'] = csv_file
        elif 'val' in filename.lower() and not 'indices' in filename and not 'scores' in filename:
            patterns['val'] = csv_file
    
    print(f"\nDetected patterns:")
    for split, path in patterns.items():
        print(f"  {split}: {os.path.basename(path)}")
    
    return patterns

def convert_kinetics_to_videomae(kinetics_csv_path, video_root_dir, output_csv_path, split='train'):
    """
    Convert Kinetics CSV to VideoMAE format.
    
    Args:
        kinetics_csv_path: Path to the original Kinetics CSV file
        video_root_dir: Root directory containing the video files
        output_csv_path: Path for the output VideoMAE format CSV
        split: 'train', 'val', or 'test'
    """
    print(f"Processing: {kinetics_csv_path}")
    
    # Read the Kinetics CSV
    df = pd.read_csv(kinetics_csv_path)
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # The CSV might already be filtered by split, but let's check
    if 'split' in df.columns:
        original_len = len(df)
        df = df[df['split'] == split]
        print(f"Filtered from {original_len} to {len(df)} rows for split '{split}'")
    
    # Get unique labels and create label to index mapping
    unique_labels = sorted(df['label'].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Save label mapping for reference
    label_mapping_path = output_csv_path.replace('.csv', '_label_mapping.txt')
    with open(label_mapping_path, 'w') as f:
        for label, idx in label_to_idx.items():
            f.write(f"{idx}: {label}\n")
    
    print(f"Found {len(unique_labels)} unique labels")
    print(f"Label mapping saved to: {label_mapping_path}")
    
    # Convert to VideoMAE format
    videomae_data = []
    missing_videos = []
    
    for _, row in df.iterrows():
        # Generate video filename (same format as original)
        video_name = f"{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}.mp4"
        video_path = os.path.join(video_root_dir, split, video_name)
        
        # Check if video exists
        if os.path.exists(video_path):
            label_idx = label_to_idx[row['label']]
            videomae_data.append([video_path, label_idx])
        else:
            missing_videos.append(video_name)
    
    # Create VideoMAE format DataFrame
    videomae_df = pd.DataFrame(videomae_data, columns=['video_path', 'label'])
    
    # Save to CSV without header (VideoMAE doesn't expect header)
    # Use space delimiter as expected by kinetics.py
    videomae_df.to_csv(output_csv_path, index=False, header=False, sep=' ')
    
    print(f"Converted {len(videomae_data)} videos for {split} split")
    print(f"Missing {len(missing_videos)} videos")
    print(f"VideoMAE format CSV saved to: {output_csv_path}")
    
    if missing_videos:
        missing_file = output_csv_path.replace('.csv', '_missing_videos.txt')
        with open(missing_file, 'w') as f:
            for video in missing_videos:
                f.write(f"{video}\n")
        print(f"Missing videos list saved to: {missing_file}")
    
    return len(videomae_data), len(missing_videos)

def main():
    parser = argparse.ArgumentParser(description='Convert Kinetics CSV annotations to VideoMAE format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing CSV annotation files')
    parser.add_argument('--video_root', type=str, default=os.path.expanduser('~/kinetics-dataset/k400'),
                        help='Root directory containing video files (default: ~/kinetics-dataset/k400)')
    parser.add_argument('--output_dir', type=str, default='./kinetics_videomae_annotations',
                        help='Output directory for VideoMAE format annotations (default: ./kinetics_videomae_annotations)')
    
    args = parser.parse_args()
    
    # Expand user paths
    args.input_dir = os.path.expanduser(args.input_dir)
    args.video_root = os.path.expanduser(args.video_root)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Converting Kinetics CSV Files to VideoMAE Format ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Video root: {args.video_root}")
    print(f"Output directory: {args.output_dir}")
    
    # Detect filename patterns
    patterns = detect_filename_patterns(args.input_dir)
    
    if not patterns:
        print("Error: No CSV files matching expected patterns found")
        print("Expected patterns:")
        print("  - Standard: train.csv, val.csv, test.csv")
        print("  - Training subset: *train*.csv, *eval*.csv, *val*.csv, *test*.csv")
        return
    
    total_found = 0
    total_missing = 0
    
    # Process each detected file
    for split, input_csv in patterns.items():
        output_csv = os.path.join(args.output_dir, f"{split}.csv")
        
        print(f"\n=== Converting {os.path.basename(input_csv)} -> {split}.csv ===")
        found, missing = convert_kinetics_to_videomae(
            input_csv, args.video_root, output_csv, split
        )
        
        if found + missing > 0:
            success_rate = 100 * found / (found + missing)
            print(f"Success rate: {found}/{found+missing} ({success_rate:.1f}%)")
            total_found += found
            total_missing += missing
        else:
            print("No videos found in this file")
    
    print(f"\n=== Summary ===")
    print(f"Total videos found: {total_found}")
    print(f"Total videos missing: {total_missing}")
    if total_found + total_missing > 0:
        overall_rate = 100 * total_found / (total_found + total_missing)
        print(f"Overall success rate: {overall_rate:.1f}%")
    
    # List all files in input directory for reference
    print(f"\n=== All files in {args.input_dir} ===")
    if os.path.exists(args.input_dir):
        for file in sorted(os.listdir(args.input_dir)):
            filepath = os.path.join(args.input_dir, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {file} ({size:,} bytes)")

if __name__ == "__main__":
    main() 