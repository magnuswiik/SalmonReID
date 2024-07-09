import torch
import os, json
import numpy as np
from torchvision.io import read_image
from torchvision import tv_tensors

class SalmonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = [file for file in sorted(os.listdir(os.path.join(root, "pipeline_images"))) if file.endswith(('.jpg', '.jpeg', '.png'))]
        self.annots = [file for file in sorted(os.listdir(os.path.join(root, "pipeline_annots"))) if file.endswith('.json')]
    
    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "Images", self.images[idx])
        annots_path = os.path.join(self.root, "Boxes", self.annots[idx])
        
        img = read_image(img_path)
        
        # Filter out DS.store file. Only relevant for MacOS
        if "DS" not in annots_path:
            try:
                shapes = json.load(open(annots_path))['shapes']
            except:
                print("Cannot open json file...")
            
        boxes = [shape['points'] for shape in shapes]
        boxes_flattened = []
        for box in boxes:
            box = np.array(box).flatten().tolist()
            boxes_flattened.append(box)
        
        boxes_flattened = torch.tensor(boxes_flattened, dtype=torch.float)
        num_objs = len(boxes_flattened)

        # there is only one class -> Salmon = 1
        labels = torch.ones((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)
        img = img.float() / 255.0
        
        target = {}
        target["boxes"] = boxes_flattened
        target["labels"] = labels
        target["image_id"] = torch.tensor(idx)
        target["individual"] = torch.tensor([shape['group_id'] for shape in shapes])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        if target["boxes"].numel() == 0:
            print("No boxes :(")

        return img, target

    def __len__(self):
        return len(self.images)