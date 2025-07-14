#!/usr/bin/env python3
"""
Filter validation and test sets to only include classes present in the training set.
This ensures consistent evaluation on the same classes the model was trained on.
"""

import pandas as pd
import os
import argparse
from pathlib import Path

def load_label_mapping(mapping_file):
    """Load label mapping from file."""
    label_to_idx = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                idx, label = line.split(': ', 1)
                label_to_idx[label] = int(idx)
    return label_to_idx

def filter_csv_by_classes(csv_path, train_classes, output_path):
    """Filter CSV file to only include classes present in training set."""
    print(f"Processing: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path, header=None, delimiter=' ')
    print(f"Original samples: {len(df)}")
    
    # Filter to only include classes in training set
    filtered_df = df[df[1].isin(train_classes)]
    print(f"Filtered samples: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} samples")
    
    # Save filtered CSV
    filtered_df.to_csv(output_path, header=False, index=False)
    print(f"Saved filtered CSV to: {output_path}")
    
    return filtered_df

def create_filtered_label_mapping(train_mapping, filtered_classes, output_path):
    """Create new label mapping for filtered classes."""
    print(f"Creating filtered label mapping: {output_path}")
    
    # Read original mapping
    with open(train_mapping, 'r') as f:
        lines = f.readlines()
    
    # Filter to only include classes in filtered set
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line:
            idx, label = line.split(': ', 1)
            if label in filtered_classes:
                filtered_lines.append(line)
    
    # Create new mapping with consecutive indices
    new_mapping = []
    for i, line in enumerate(filtered_lines):
        idx, label = line.split(': ', 1)
        new_mapping.append(f"{i}: {label}")
    
    # Save new mapping
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_mapping))
    
    print(f"Created mapping with {len(new_mapping)} classes")
    return new_mapping

def main():
    parser = argparse.ArgumentParser(description='Filter validation and test sets to match training classes')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train.csv, val.csv, test.csv and label mappings')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for filtered files (default: data_dir/filtered)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'filtered'
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load training classes
    train_mapping = data_dir / 'train_label_mapping.txt'
    if not train_mapping.exists():
        print(f"Error: {train_mapping} not found")
        return
    
    train_label_to_idx = load_label_mapping(train_mapping)
    train_classes = set(train_label_to_idx.values())
    print(f"Training set has {len(train_classes)} classes")
    
    # Filter validation set
    val_csv = data_dir / 'val.csv'
    if val_csv.exists():
        filtered_val_csv = output_dir / 'val.csv'
        filtered_val_df = filter_csv_by_classes(val_csv, train_classes, filtered_val_csv)
        
        # Create filtered validation label mapping
        val_classes = set(filtered_val_df[1].unique().tolist())
        create_filtered_label_mapping(train_mapping, val_classes, output_dir / 'val_label_mapping.txt')
    
    # Filter test set
    test_csv = data_dir / 'test.csv'
    if test_csv.exists():
        filtered_test_csv = output_dir / 'test.csv'
        filtered_test_df = filter_csv_by_classes(test_csv, train_classes, filtered_test_csv)
        
        # Create filtered test label mapping
        test_classes = set(filtered_test_df[1].unique().tolist())
        create_filtered_label_mapping(train_mapping, test_classes, output_dir / 'test_label_mapping.txt')
    
    # Copy training files
    import shutil
    train_csv = data_dir / 'train.csv'
    train_mapping_file = data_dir / 'train_label_mapping.txt'
    
    if train_csv.exists():
        shutil.copy2(train_csv, output_dir / 'train.csv')
        print(f"Copied training CSV to: {output_dir / 'train.csv'}")
    
    if train_mapping_file.exists():
        shutil.copy2(train_mapping_file, output_dir / 'train_label_mapping.txt')
        print(f"Copied training label mapping to: {output_dir / 'train_label_mapping.txt'}")
    
    print("\n✅ Filtering complete!")
    print(f"Filtered dataset available in: {output_dir}")
    print(f"Use --data_path {output_dir} --nb_classes {len(train_classes)} for training")

if __name__ == "__main__":
    main() 