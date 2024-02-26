import os
from utils import visualize_preds, visualize_annot, pred_to_labelme
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
from tqdm.auto import tqdm

def get_detection_model(num_classes, weights=FasterRCNN_ResNet50_FPN_Weights):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def inference(datapath, device):
    
    num_classes = 2

    model = get_detection_model(num_classes)
    
    model.load_state_dict(torch.load('/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/models/model2/model1.pt', map_location=device))

    model.to(device)
    
    model.eval()
    
    data = sorted(os.listdir(datapath))
    
    prog_bar = tqdm(data, total=len(data))
    
    for i, filename in enumerate(prog_bar):
        if filename.endswith(".jpg") or filename.endswith(".png"):
 
            image_path = os.path.join(datapath, filename)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = np.transpose(image / 255.0, (2, 0, 1)) # Pixel values need to be between 0 and 1 and transpose image 
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and add to device

            with torch.no_grad():
                outputs = model(image_tensor)
                
            #visualize_preds(image_tensor, outputs)
            pred_to_labelme(outputs, image_path)
            #visualize_annot(image_path)

