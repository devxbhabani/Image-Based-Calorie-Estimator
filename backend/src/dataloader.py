import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from src import config

class NutritionDataset(Dataset):
    def __init__(self, metadata_path, imagery_dir, transform=None):
        self.imagery_dir = imagery_dir
        self.transform = transform

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        raw_df = pd.read_csv(metadata_path)

        if 'ingr_name' in raw_df.columns:
            self.df = raw_df.groupby('dish_id').agg({
                'calories': 'sum',
                'grams': 'sum'
            }).reset_index()
        else:
            self.df = raw_df
            
        # Filter valid records with available images
        self.valid_indices = []
        for idx in range(len(self.df)):
            dish_id = self.df.iloc[idx, 0]
            img_path = os.path.join(self.imagery_dir, f"{dish_id}.png")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
                
        print(f"Loaded {len(self.valid_indices)} valid records out of {len(self.df)} total dishes.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        
        # Target column parsing
        dish_id = row['dish_id']
        calories = float(row['calories'])
        mass = float(row['grams'])
        
        img_path = os.path.join(self.imagery_dir, f"{dish_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        targets = torch.tensor([calories, mass], dtype=torch.float32)
        return image, targets

        