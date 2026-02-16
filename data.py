#----------------------------------------------------------
# Copyright (C) 2025 Etore Maloso Tronconi
# Embedded Systems & Instrumentation Department
# École Supérieure d'Ingénieurs - ESIGELEC, France
# This software is intended for research purposes only;
# its redistribution is forbidden under any circumstances.
#----------------------------------------------------------
import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset

#==========================================================

class FramePictures(Dataset):
    def __init__(self, data_path, noise=None, nfold=0, load_first=False, num_frames=4):
        """
        Dataset for loading sequences of frames with corresponding labels.
        
        Args:
            data_path: Path to the dataset directory
            data_type: 'train' or 'test'
            noise: Noise level for data augmentation (optional)
            nfold: Fold number for cross-validation
            load_first: Whether to preload all data
            num_frames: Number of consecutive frames to load (default: 4)
        """
        self.noise = noise
        self.nfold = nfold
        self.load_first = load_first
        self.num_frames = num_frames
        
        self.image_dir = os.path.join(data_path, 'images')
        self.label_dir = os.path.join(data_path, 'labels')
        
        all_images = glob.glob(os.path.join(self.image_dir, '*'))
        all_images.sort(key=lambda x: self._extract_file_info(x))
        
        self.sample_groups = self._group_by_sample(all_images)
        
        
        self.sequences = self._create_sequences()
        
        if self.load_first:
            self._preload_data()

    def _extract_file_info(self, filepath):
        """Extract sample ID and frame number from filename."""
        basename = os.path.basename(filepath)
        parts = basename.split('_')
        try:
            sample_id = int(parts[1])
            frame_num = int(parts[3].split('.')[0])
            return (sample_id, frame_num)
        except (IndexError, ValueError):
            return (0, 0)

    def _group_by_sample(self, image_list):
        """Group images by sample ID."""
        groups = {}
        for img_path in image_list:
            sample_id, frame_num = self._extract_file_info(img_path)
            if sample_id not in groups:
                groups[sample_id] = []
            groups[sample_id].append((frame_num, img_path))
        
        # Sort frames within each sample
            groups[sample_id].sort(key=lambda x: x[0])
        
        return groups
    
    def _create_sequences(self):
        """Create sequences of consecutive frames."""
        sequences = []
        
        for sample_id, frames in self.sample_groups.items():
            for i in range(len(frames) - self.num_frames + 1):
                sequence = {
                    'sample_id': sample_id,
                    'frames': [frames[i + j][1] for j in range(self.num_frames)],
                    'label_path': self._get_label_path(sample_id)
                }
                sequences.append(sequence)
        
        return sequences

    def _get_label_path(self, sample_id):
        """Get the label path for a given sample ID."""
        label_pattern = os.path.join(self.label_dir, f'*_{sample_id:03d}.*')
        label_files = glob.glob(label_pattern)
        
        if len(label_files) > 0:
            return label_files[0]
        else:
            label_pattern = os.path.join(self.label_dir, f'*_{sample_id}.*')
            label_files = glob.glob(label_pattern)
            return label_files[0] if label_files else None

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        rand = random.random()

        if not self.load_first:
            frames = []
            for frame_path in sequence['frames']:
                frame = self.load_image(frame_path, rand, 'image')
                if frame.ndim == 2:
                    frame = frame[np.newaxis, ...]  
                frames.append(frame)

            frames = np.stack(frames, axis=1)
            label = self.load_image(sequence['label_path'], rand, 'label')
           
            if label.ndim == 2:
                label = label[np.newaxis, ...]  

            return torch.from_numpy(frames).float(), torch.from_numpy(label).float()
        else:
            return self.preloaded_frames[idx], self.preloaded_labels[idx]

    def _preload_data(self):
        """Preload all data into memory."""
        self.preloaded_frames = []
        self.preloaded_labels = []
        
        random_numbers = [random.uniform(0.0, 1.0) for _ in range(len(self.sequences))]
        
        for idx, sequence in enumerate(self.sequences):
            rand = random_numbers[idx]
            
            frames = []
            for frame_path in sequence['frames']:
                frame = self.load_image(frame_path, rand, 'image')
                frames.append(frame)
            frames = np.stack(frames, axis=1)
            
            label = self.load_image(sequence['label_path'], rand, 'label')
            
            self.preloaded_frames.append(frames)
            self.preloaded_labels.append(label)

    
    def load_image(self, filename, rand, image_type='label'):
        """Load and preprocess image."""
        if filename is None:
            raise ValueError(f"Label file not found")
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


        if self.noise is not None and rand > 0.5:
            image = np.flip(image, axis=1).copy()

        image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = image.astype(np.float32) / 255.0

        if self.noise is not None and image_type == 'image':
            sigma = np.random.uniform(high=self.noise)
            noise = np.random.normal(scale=sigma, size=image.shape)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        return image

#----------------------------------------------------------

if __name__ == '__main__':
    # Example usage
    data_path = "D:\\etore\\Code\\MATLAB\\SquaresCircle\\HybridUnetDataset_Cover\\train"
    
    train_data = FramePictures(
        data_path, 
        noise=0.025, 
        nfold=1, 
        load_first=False,
        num_frames=4
    )
    
    test_data = FramePictures(
        data_path, 
        nfold=1, 
        load_first=False,
        num_frames=4
    )
    
    print(f"Number of training sequences: {len(train_data)}")
    print(f"Number of test sequences: {len(test_data)}")
    
    if len(train_data) > 0:
        frames, label = train_data[0]
        print(f"\nSample data shapes:")
        print(f"Frames shape: {frames.shape}")  
        print(f"Label shape: {label.shape}")    
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(4):
            axes[i].imshow(frames[i], cmap='gray')
            axes[i].set_title(f'Frame {i+1}')
            axes[i].axis('off')
        axes[4].imshow(label[0], cmap='gray')
        axes[4].set_title('Label')
        axes[4].axis('off')
        plt.tight_layout()
        plt.savefig('sample_sequence.png')
        print("\nVisualization saved to 'sample_sequence.png'")
