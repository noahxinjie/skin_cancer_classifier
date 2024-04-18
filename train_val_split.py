import os
import shutil
import numpy as np

SEED = 4 # standardise seed for reproducibility
VAL_SIZE = 0.2

def create_validation_set(train_dir, val_size=VAL_SIZE):
    classes = os.listdir(train_dir)
    for cls in classes:
        class_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(class_dir):  # Skip non-directory files
            continue
       
        # Create directory for the validation set
        val_class_dir = os.path.join(train_dir, '..', 'val', cls)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all files from the class directory
        files = os.listdir(class_dir)
        np.random.seed(SEED) 
        np.random.shuffle(files)  # Shuffle the files to ensure random split
        num_val_files = int(len(files) * val_size)  # Number of validation files

        # Split files for validation
        val_files = files[:num_val_files]

        # Move the files to the validation directory
        for f in val_files:
           shutil.move(os.path.join(class_dir, f), os.path.join(val_class_dir, f))

train_dir = 'isic2019_modified/train'  # Update this path to your train directory
create_validation_set(train_dir)