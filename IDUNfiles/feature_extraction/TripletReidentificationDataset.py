import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import os, json, re, random
import torchvision.transforms.functional as TF
import PIL 


class TripletReidentificationDataset(Dataset):
    def __init__(self, data_path, landmark, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.landmark = landmark
        self.images = []
        self.targets = []
        for file in os.listdir(data_path):
            if not file.startswith('.'):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(file)
                    fish_id = [int(x) for x in re.findall(r'\d+', file)][0]
                    self.targets.append(fish_id)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        # Random seed
        random.seed(0)
        
        to_pil = transforms.ToPILImage()
        
        # Anchor
        anchor_label = torch.tensor(self.targets[idx])
        anchor_img_path = os.path.join(self.data_path, self.images[idx][:self.images[idx].find("_", 0, 7)], self.landmark, self.images[idx])
        anchor_img = to_pil(read_image(anchor_img_path) / 255.0)
        
        # Positive
        positive_idx = random.choice([i for i, label in enumerate(self.targets) if label == anchor_label and i != idx])
        positive_label = torch.tensor(self.targets[positive_idx])
        positive_img_path = os.path.join(self.data_path,self.images[positive_idx][:self.images[positive_idx].find("_", 0, 7)], self.landmark, self.images[positive_idx])
        positive_img = to_pil(read_image(positive_img_path) / 255.0)
        
        # Negative
        negative_idx = random.choice([i for i, label in enumerate(self.targets) if label != anchor_label])
        negative_label = torch.tensor(self.targets[negative_idx])
        negative_img_path = os.path.join(self.data_path, self.images[negative_idx][:self.images[negative_idx].find("_", 0, 7)], self.landmark, self.images[negative_idx])
        negative_img = to_pil(read_image(negative_img_path) / 255.0)
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        imgs = [anchor_img, positive_img, negative_img]
        targets = [anchor_label, positive_label, negative_label]
            
        return imgs, targets