import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os, json
import torchvision.transforms.functional as TF

class BodypartsDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.annots = []
        for folder in os.listdir(data_path):
            if (not folder.startswith('.')) and os.path.isdir(os.path.join(data_path, folder)):
                for file in sorted(os.listdir(os.path.join(data_path, folder))):
                    if file.endswith('.json'):
                        self.annots.append(file)
        for file in self.annots:
            file_jpg = file[:-4] + "jpg"
            self.images.append(file_jpg)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        labels_map = {'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5} # Husk å fjerne pectoral og fikse datasett!

        img_path = os.path.join(self.data_path, self.images[idx][:self.images[idx].find("_",0, 7)], self.images[idx])
        annot_path = os.path.join(self.data_path, self.annots[idx][:self.annots[idx].find("_",0, 7)], self.annots[idx])

        img = read_image(img_path) / 255.0
        annot = json.load(open(annot_path))
        boxes = [[shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]] for shape in annot['shapes'] if shape['shape_type'] == 'rectangle']
        labels = [labels_map[shape['label']] for shape in annot['shapes'] if shape['shape_type'] == 'rectangle']
        
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes,
                  'labels': labels,}
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target