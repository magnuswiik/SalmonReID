import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os, json, re, random
import torchvision.transforms.functional as TF


class ReidentificationDataset(Dataset):
    def __init__(self, data_path, landmark, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.landmark = landmark
        self.images = []
        self.targets = []
        for fish in os.listdir(data_path):
            if not fish.startswith('.'):
                for folder in os.listdir(os.path.join(data_path, fish)):
                    if (not folder.startswith('.')) and folder == landmark:
                        for file in sorted(os.listdir(os.path.join(data_path, fish, folder))):
                            if file.endswith(('.jpg', '.jpeg', '.png')):
                                self.images.append(file)
                                fish_id = [int(x) for x in re.findall(r'\d+', file)][0]
                                self.targets.append(fish_id)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        # Random seed
        random.seed(0)
        
        # Anchor
        anchor_label = torch.tensor(self.targets[idx])
        anchor_img_path = os.path.join(self.data_path, self.images[idx][:self.images[idx].find("_", 0, 7)], self.landmark, self.images[idx])
        anchor_img = read_image(anchor_img_path) / 255.0
        
        # Positive
        positive_idx = random.choice([i for i, label in enumerate(self.targets) if label == anchor_label and i != idx])
        positive_label = torch.tensor(self.targets[positive_idx])
        positive_img_path = os.path.join(self.data_path,self.images[positive_idx][:self.images[positive_idx].find("_", 0, 7)], self.landmark, self.images[positive_idx])
        positive_img = read_image(positive_img_path) / 255.0
        
        # Negative
        negative_idx = random.choice([i for i, label in enumerate(self.targets) if label != anchor_label])
        negative_label = torch.tensor(self.targets[negative_idx])
        negative_img_path = os.path.join(self.data_path, self.images[negative_idx][:self.images[negative_idx].find("_", 0, 7)], self.landmark, self.images[negative_idx])
        negative_img = read_image(negative_img_path) / 255.0
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        imgs = [anchor_img.unsqueeze(0), positive_img.unsqueeze(0), negative_img.unsqueeze(0)]
        targets = [anchor_label, positive_label, negative_label]
            
        return imgs, targets