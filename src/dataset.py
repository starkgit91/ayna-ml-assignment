import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class PolygonDataset(Dataset):
    """
    Dataset for polygon images.
    Handles paired transformations for data augmentation.
    """
    def __init__(self, root_dir, image_size=128, is_train=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train  # Flag to control augmentation
        self.data = json.load(open(os.path.join(root_dir, 'data.json')))
        
        # FIXED: Added 'magenta' to the list of all possible colors.
        all_colors = sorted(['blue', 'cyan', 'green', 'orange', 'pink', 'purple', 'red', 'yellow', 'magenta'])
        self.color_to_idx = {color: i for i, color in enumerate(all_colors)}
        self.colors = all_colors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_image_path = os.path.join(self.root_dir, 'inputs', item['input_polygon'])
        output_image_path = os.path.join(self.root_dir, 'outputs', item['output_image'])
        
        input_image = Image.open(input_image_path).convert('RGB')
        output_image = Image.open(output_image_path).convert('RGB')

        # --- Paired Transformations ---
        # 1. Resize both images
        resize = transforms.Resize((self.image_size, self.image_size))
        input_image = resize(input_image)
        output_image = resize(output_image)

        # 2. Apply augmentations only to the training set
        if self.is_train:
            # Random horizontal flip (applied to both)
            if random.random() > 0.5:
                input_image = transforms.functional.hflip(input_image)
                output_image = transforms.functional.hflip(output_image)

            # Random rotation (applied to both)
            angle = random.uniform(-30, 30)
            input_image = transforms.functional.rotate(input_image, angle)
            output_image = transforms.functional.rotate(output_image, angle)

        # 3. Convert both to tensor
        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(input_image)
        output_tensor = to_tensor(output_image)
        
        # 4. Normalize both tensors to [-1, 1] range
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = normalize(input_tensor)
        output_tensor = normalize(output_tensor)
        
        # Get color information
        color_name = item['colour']
        color_idx = self.color_to_idx[color_name]
        
        return input_tensor, torch.tensor(color_idx), output_tensor
