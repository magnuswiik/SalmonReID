import torch
import os, json
import numpy as np
from torchvision.io import read_image
from torchvision import tv_tensors

class SalmonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        
        self.imgs = [file for file in sorted(os.listdir(os.path.join(root, "Images"))) if file.endswith(('.jpg', '.jpeg', '.png'))]
        self.annots = [file for file in sorted(os.listdir(os.path.join(root, "Boxes"))) if file.endswith('.json')]
    
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        annots_path = os.path.join(self.root, "Boxes", self.annots[idx])
        
        img = read_image(img_path)
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
        

        # Number of boxes in image
        num_objs = len(boxes_flattened)

        # there is only one class -> Salmon = 1
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)
        img = img.float() / 255.0
        

        target = {}
        target["boxes"] = boxes_flattened
        target["labels"] = labels
        target["image_id"] = idx
        target["area"] = (boxes_flattened[:,2]-boxes_flattened[:,0])*(boxes_flattened[:,3]-boxes_flattened[:,1])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        if target["boxes"].numel() == 0:
            print("No boxes :(")

        return img, target

    def __len__(self):
        return len(self.imgs)