import os
import shutil
import random
from pathlib import Path

def split_data(source_dir, train_dir, val_dir, val_ratio=0.1, seed=42):
    """Split data files into training and validation sets.
    
    Args:
        source_dir (str): Directory containing the source data files
        train_dir (str): Directory to store training files
        val_dir (str): Directory to store validation files
        val_ratio (float): Ratio of files to use for validation (0-1)
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all unique prefixes (e.g., 'arxiv', 'book', etc.)
    files = os.listdir(source_dir)
    prefixes = set()
    for f in files:
        if f.endswith('.bin'):
            prefix = f.split('_sample_')[0]
            prefixes.add(prefix)
    
    # For each prefix, split files maintaining the same ratio
    for prefix in prefixes:
        prefix_files = [f for f in files if f.startswith(prefix)]
        val_size = max(1, int(len(prefix_files) * val_ratio))
        
        # Randomly select validation files
        val_files = random.sample(prefix_files, val_size)
        train_files = [f for f in prefix_files if f not in val_files]
        
        # Copy files to respective directories
        for f in train_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(train_dir, f)
            )
        
        for f in val_files:
            shutil.copy2(
                os.path.join(source_dir, f),
                os.path.join(val_dir, f)
            )
        
        print(f"{prefix}: {len(train_files)} train, {len(val_files)} validation files")

if __name__ == "__main__":
    source_dir = "data/redpajama_sample"
    train_dir = "data/redpajama_train"
    val_dir = "data/redpajama_val"
    
    # Create directories if they don't exist
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    
    # Split with 10% validation data
    split_data(source_dir, train_dir, val_dir, val_ratio=0.1)
