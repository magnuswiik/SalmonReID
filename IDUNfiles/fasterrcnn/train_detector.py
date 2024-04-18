import os
import utils
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
from SalmonDataset import SalmonDataset
from LandmarksDataset import LandmarksDataset
import torchvision
from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
plt.style.use('ggplot')
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def get_detection_model(num_classes, weights=FasterRCNN_ResNet50_FPN_Weights):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def get_mobile_detection_model(num_classes, weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights):

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def make_datasets(datapath):
    
    # Random seed for reproducibility
    random.seed(0)

    # use our dataset and defined transformations
    dataset = SalmonDataset(datapath)
    dataset_validation = SalmonDataset(datapath)
    dataset_test = SalmonDataset(datapath)

    data_indices = np.arange(0,len(dataset.imgs), dtype=np.int16).tolist()

    indices_test = random.sample(data_indices, int(len(data_indices)*0.2))
    data_indices = [idx for idx in data_indices if idx not in indices_test]

    indices_validation = random.sample(data_indices, int(len(data_indices)*0.2))
    data_indices = [idx for idx in data_indices if idx not in indices_validation]

    indices_training = random.sample(data_indices, int(len(data_indices)))

    # split the dataset in train and test set
    dataset_training = torch.utils.data.Subset(dataset, indices_training) # 80% for training and validation
    dataset_validation = torch.utils.data.Subset(dataset_validation, indices_validation)
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test) # 20% for testing
    
    return dataset_training, dataset_validation, dataset_test

def train_landmarks(datapath, epochs, lr, device):
    
    dataset = LandmarksDataset(datapath)
    
     # Random seed for reproducibility
    g = torch.manual_seed(0)
    
    # our dataset has two classes only - background and landmarks
    num_classes = 6

    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g
    )
    
    # get the model using our helper function
    model = get_detection_model(num_classes)

    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    
    ### Training

    train_loss_list = []
    lr_step_sizes = []

    for epoch in range(epochs):
        
        # initialize tqdm progress bar
        prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
        
        train_loss_per_epoch = []

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move target tensors to the same device
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_per_epoch.append(loss_value)
            losses.backward()
            optimizer.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"|Epoch: {epoch+1}/{epochs}| Loss: {loss_value:.4f}")
   
        # Save metrics per epoch
        lr_step_sizes.append(optimizer.param_groups[0]['lr'])
        train_loss_list.append(sum(loss for loss in train_loss_per_epoch)/len(train_loss_per_epoch))
        
        # Change step size in optimizer
        lr_scheduler.step()


    ### SAVING RESULTS
    
    cwd = os.getcwd()
    MODELPATH = cwd + "/landmark_models/model1/"

    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
        
    dict = {'training_loss': train_loss_list, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(MODELPATH + 'metrics.csv', index=False)

    torch.save(model.state_dict(), MODELPATH + "model1.pt")
    print("Model is saved at:" + MODELPATH + "model1.pt")

def train(datapath, epochs, lr, device):
    
    # Create directory for model
    cwd = os.getcwd()
    MODELPATH = cwd + "/models/fasterrcnn/"

    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
    
    dataset_training, dataset_validation, dataset_test = make_datasets(datapath)
    
    # Random seed for reproducibility
    g = torch.manual_seed(0)
    
    # our dataset has two classes only - background and salmon
    num_classes = 2

    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(
        dataset_training,
        batch_size=5,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn,
        generator=g
    )

    # define training and validation data loaders
    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=3,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn,
        generator=g
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g
    )

    # get the model using our helper function
    #model = get_mobile_detection_model(num_classes)
    model = get_detection_model(num_classes)

    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.2, patience=10
)

    ### Training

    train_loss_hist = Averager()
    train_loss_list = []
    lr_step_sizes = []
    validation_losses = []

    best_validation_loss = float('inf')  # Initialize with a large value

    for epoch in range(epochs):
        
        # initialize tqdm progress bar
        prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
        
        train_loss_per_epoch = []

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move target tensors to the same device
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_per_epoch.append(loss_value)
            losses.backward()
            optimizer.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"|Epoch: {epoch+1}/{epochs}| Loss: {loss_value:.4f}")
        
        validation_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader_validation:
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                validation_loss += loss.item()
        
        
        # Save metrics per epoch
        lr_step_sizes.append(optimizer.param_groups[0]['lr'])
        train_loss_list.append(sum(loss for loss in train_loss_per_epoch)/len(train_loss_per_epoch))
        train_loss_hist.send(sum(loss for loss in train_loss_per_epoch)/len(train_loss_per_epoch))
        validation_loss /= len(data_loader_validation)
        validation_losses.append(validation_loss)
        
        # Save the model if it has the best validation loss
        if validation_loss < best_validation_loss and epoch > 10:
            best_validation_loss = validation_loss
            best_model_path = os.path.join(MODELPATH, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            #print(f"Best model saved at: {best_model_path}")
        
        # Change step size in optimizer
        lr_scheduler.step(validation_loss)


    ### SAVING RESULTS
        
    dict = {'training_loss': train_loss_list,  'validation_loss': validation_losses, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(MODELPATH + 'metrics.csv', index=False)


def test(datapath, modelpath, device):
    
    classes = {0: 'background',
               1: 'salmon'}
    
    model = get_detection_model(num_classes=2)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.eval().to(device)
    
    dataset_training, dataset_validation, dataset_test = make_datasets(datapath)
    
    # Random seed for reproducibility
    g = torch.manual_seed(0)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g
    )
    
    prog_bar = tqdm(data_loader_test, total=len(data_loader_test))
    
    targets_total = []
    preds_total = []
    
    metric = MeanAveragePrecision(iou_type="bbox")

    for i, data in enumerate(prog_bar):
        
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move target tensors to the same device
        with torch.no_grad():
            preds = model(images)
        
        metric.update(preds, targets)
        
    pprint(metric.compute())

    #utils.visualize_preds(images, preds)