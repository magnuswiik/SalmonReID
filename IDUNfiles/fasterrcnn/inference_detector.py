import os, time
from utils import visualize_preds, visualize_annot, pred_to_labelme
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
import numpy as np
import cv2
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

def filter_landmark_predictions(predictions):
    
    labels_map = {1:'tailfin', 2:'dorsalfin', 3:'thorax', 4:'pectoralfin', 5:'eyeregion'}
    
    boxes = predictions['boxes'].tolist()
    labels = predictions['labels'].tolist()
    scores = predictions['scores'].tolist()
    
    unique_boxes = []
    unique_labels = []
    unique_scores = []
    seen_labels = set()
    
    for box, label, score in zip(boxes, labels, scores):
        landmark = labels_map[label]
        
        if label not in seen_labels:
            unique_boxes.append(box)
            unique_labels.append(label)
            unique_scores.append(score)
            seen_labels.add(label)

        '''if landmark == 'dorsalfin':
            dorsal_y1 = box[1]
            pectoral_y1 = boxes[labels.index(4)][1]
            if not (dorsal_y1 < pectoral_y1):
                boxes.pop(i)
                labels.pop(i)
                scores.pop(i)'''
    
    filtered_predictions = {'boxes':torch.Tensor(unique_boxes), 'labels':torch.Tensor(unique_labels), 'scores': torch.Tensor(unique_scores)}
    
    return filtered_predictions
    
def crop_landmark_predictions(img, img_name, fish, predictions):
    
    to_path = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Prediksjoner_Identifikasjonssett/"
    
    labels_map = {1:'tailfin', 2:'dorsalfin', 3:'thorax', 4:'pectoralfin', 5:'eyeregion'}
    
    img = np.transpose(img*255, (1, 2, 0)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = predictions['boxes'].tolist()
    labels = predictions['labels'].tolist()
    
    for box, label in zip(boxes, labels):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        landmark = labels_map[label]
        cropped_image = img[y1:y2, x1:x2, :]
        
        if not os.path.isdir(to_path + fish + "/" + landmark):
            os.makedirs(to_path + fish + "/" + landmark)
            
        filename = to_path + fish + "/" + landmark + "/" + fish + "_" + landmark + img_name[img_name.find("_"):]
        cv2.imwrite(filename, cropped_image)
    

def get_detection_model(num_classes, weights=FasterRCNN_ResNet50_FPN_Weights):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def get_mobile_detection_model(num_classes, weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights):

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def inference(datapath, modelpath, device):
    
    num_classes = 2

    model = get_mobile_detection_model(num_classes)
    
    model.load_state_dict(torch.load(modelpath, map_location=device))

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

            start = time.time()
            with torch.no_grad():
                outputs = model(image_tensor)
            end = time.time()
            print("Prediction time: ", end - start)  
            visualize_preds("salmon", image_tensor, outputs)
            #pred_to_labelme(outputs, image_path)
            #visualize_annot(image_path)

def inference_landmark(datapath, modelpath, device):
    
    num_classes = 6

    model = get_detection_model(num_classes)
    
    model.load_state_dict(torch.load(modelpath, map_location=device))

    model.to(device)
    
    model.eval()
    
    data = sorted(os.listdir(datapath))
    
    prog_bar = tqdm(data, total=len(data))
    
    for i, folder in enumerate(prog_bar):
        if (not folder.startswith('.')) and os.path.isdir(os.path.join(datapath, folder)) and folder != 'old':
            for file in sorted(os.listdir(os.path.join(datapath, folder))):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(datapath, folder, file)
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    image = np.transpose(image / 255.0, (2, 0, 1)) # Pixel values need to be between 0 and 1 and transpose image 
                    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and add to device

                    start = time.time()
                    with torch.no_grad():
                        outputs = model(image_tensor)
                    end = time.time()
                    #print("Prediction time: ", end - start)
                    filtered_predictions = filter_landmark_predictions(outputs[0])
                    crop_landmark_predictions(image, file, folder, filtered_predictions)
                    #visualize_preds("landmarks", image_tensor[0], filtered_predictions)
                    #pred_to_labelme(outputs, image_path)
                    #visualize_annot(image_path)