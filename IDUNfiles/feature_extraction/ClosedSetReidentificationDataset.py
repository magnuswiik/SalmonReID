import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import os, json, re, random
import torchvision.transforms.functional as TF
import PIL 


class ClosedSetReidentificationDataset(Dataset):
    def __init__(self, data_path, landmark, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.landmark = landmark
        self.images = []
        self.targets = []
        self.classes = {
                        2:0,
                        5:1,
                        6:2,
                        7:3,
                        9:4,
                        12:5,
                        14:6
                    }
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
        
        to_pil = transforms.ToPILImage()
        
        # Anchor
        anchor_label = torch.tensor(self.classes[self.targets[idx]])
        anchor_img_path = os.path.join(self.data_path, self.images[idx][:self.images[idx].find("_", 0, 7)], self.landmark, self.images[idx])
        anchor_img = read_image(anchor_img_path) / 255.0
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            
        img = anchor_img
        target = anchor_label
            
        return img, target