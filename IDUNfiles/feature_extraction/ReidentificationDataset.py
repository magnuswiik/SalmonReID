import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import os, json, re, random
import torchvision.transforms.functional as TF
import PIL 


class ReidentificationDataset(Dataset):
    def __init__(self, data_path, bodypart, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.bodypart = bodypart
        self.images = []
        self.targets = []
        for file in sorted(os.listdir(data_path)):
            file_bodypart = file.split("_")[1]
            if not file.startswith('.') and file_bodypart == bodypart:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(file)
                    fish_id = [int(x) for x in re.findall(r'\d+', file)][0]
                    self.targets.append(fish_id)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        label = torch.tensor(self.targets[idx])
        img_path = os.path.join(self.data_path, self.images[idx])
        img = read_image(img_path) / 255.0
        
        if self.transform:
            img = self.transform(img)

        return img, label