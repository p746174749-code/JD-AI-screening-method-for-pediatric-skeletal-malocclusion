import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, image_names=None, transform=None):
        """
        root_dir: Root directory, organized as root/class_x/sample_y/
        image_names: List of image indices to load (e.g., [1, 2, 4])
        transform: Image preprocessing/augmentation
        """
        self.root_dir = root_dir
        self.transform = transform

        # Convert [1, 2, 4] to ['1.jpg', '2.jpg', '4.jpg']
        self.image_names = [f"{i}.jpg" for i in (image_names if image_names else [4])]
        self.samples = []

        # Iterate through class folders
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Iterate through sample subfolders
            for subfolder in sorted(os.listdir(class_dir)):
                subfolder_path = os.path.join(class_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                image_paths = []
                for name in self.image_names:
                    path = self._find_image_by_name(subfolder_path, name)
                    image_paths.append(path)

                # Add sample only if all specified images exist
                if all(image_paths):
                    # Extract label from class name (assuming format like 'name_label')
                    label = int(class_name.split('_')[-1])
                    self.samples.append((image_paths, label))

    def _find_image_by_name(self, folder, target_name):
        """ Helper to find a file in a folder matching the target name (case-insensitive) """
        for file in os.listdir(folder):
            if file.lower() == target_name.lower():
                return os.path.join(folder, file)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]
        images = []

        for path in image_paths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Return single image for 1 index, or a tuple for multiple indices
        return (images[0] if len(images) == 1 else tuple(images)), label