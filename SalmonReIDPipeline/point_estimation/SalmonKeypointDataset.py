import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os, json
import torchvision.transforms.functional as TF

class SalmonKeypointDataset(Dataset):
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
                    elif file.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(file)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        labels_map = {'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5, 'pectoral': 4} # Husk å fjerne pectoral og fikse datasett!

        img_path = os.path.join(self.data_path, self.images[idx][:self.images[idx].find("_",0, 7)], self.images[idx])
        annot_path = os.path.join(self.data_path, self.annots[idx][:self.annots[idx].find("_",0, 7)], self.annots[idx])

        img = read_image(img_path) / 255.0
        annot = json.load(open(annot_path))
        keypoints = [[point[0], point[1], 1] for shape in annot['shapes'] for point in shape['points'] if shape['shape_type'] == 'point'] # Visibility is set to 1!
        #boxes = [[shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]] for shape in annot['shapes'] if shape['shape_type'] == 'rectangle']
        #labels = [labels_map[shape['label']] for shape in annot['shapes'] if shape['shape_type'] == 'rectangle']
        boxes = [[0,0,img.shape[2], img.shape[1]]]
        labels = [1]
        
        keypoints = torch.tensor(keypoints, dtype=torch.float)
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes,
                  'labels': labels,
                  'keypoints': keypoints.unsqueeze(0)}
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        
        original_size = torch.tensor([img.shape[2], img.shape[1]])  # original image size (width, height)
        
        # Resize the image
        img = TF.resize(img, self.size)
        new_size = torch.tensor(self.size[::-1])  # target size (width, height)
        
        # Calculate scaling factors for width and height
        scale_factors = new_size / original_size
        
        # Scale bounding boxes
        target['boxes'] *= torch.cat([scale_factors, scale_factors])
        
        # Scale keypoints
        target['keypoints'][:,:, 0:2] *= scale_factors
        
        return img, target