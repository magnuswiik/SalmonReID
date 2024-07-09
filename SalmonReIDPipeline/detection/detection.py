from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from SalmonDataset import SalmonDataset
from BodypartsDataset import BodypartsDataset
import os, random
import utils
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import torchvision
from pprint import pprint

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_detection_model(num_classes, weights=FasterRCNN_ResNet50_FPN_Weights):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Add custom box predictor head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def make_datasets(datapath, task):
    
    # Random seed for reproducibility
    random.seed(0)

    if task == "salmon":
        dataset = SalmonDataset(datapath)
        dataset_validation = SalmonDataset(datapath)
        dataset_test = SalmonDataset(datapath)
    if task == "bodyparts":
        dataset = BodypartsDataset(datapath)
        dataset_validation = BodypartsDataset(datapath)
        dataset_test = BodypartsDataset(datapath)

    data_indices = np.arange(0,len(dataset.images), dtype=np.int16).tolist()

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

def train_bodypart_detection_model(datapath, epochs, lr, device):
    
    cwd = os.getcwd()
    MODELPATH = cwd + "/bodypart_detection_models/"
    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
    
    dataset_training, dataset_validation, _ = make_datasets(datapath, "bodyparts")
    
     # Random seed for reproducibility
    g = torch.manual_seed(0)

    # define training and validation data loaders
    data_loader_training = torch.utils.data.DataLoader(
        dataset_training,
        batch_size=10,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn,
        generator=g
    )
    
    data_loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=10,
        shuffle=True,
        num_workers=5,
        collate_fn=collate_fn,
        generator=g
    )
    
    # Create detection model for bodyparts detection: num_classes = background + bodyparts
    num_classes = 6
    model = get_detection_model(num_classes)

    # move model to current device
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    
    ### Training

    train_loss_list = []
    lr_step_sizes = []
    validation_losses = []

    best_validation_loss = float('inf')

    for epoch in range(epochs):
        
        # initialize tqdm progress bar
        prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
        
        train_loss_per_epoch = []

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_per_epoch.append(loss_value)
            losses.backward()
            optimizer.step()

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
        validation_loss /= len(data_loader_validation)
        validation_losses.append(validation_loss)
        
        # Save the model if it has the best validation loss
        if validation_loss < best_validation_loss and epoch > 10:
            best_validation_loss = validation_loss
            best_model_path = os.path.join(MODELPATH, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
        
        # Change step size in optimizer
        lr_scheduler.step()

    ### SAVING RESULTS

    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
        
    dict = {'training_loss': train_loss_list, 'validation_loss': validation_losses, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(MODELPATH + 'metrics.csv', index=False)

    torch.save(model.state_dict(), MODELPATH + "best_model.pt")
    print("Model is saved at:" + MODELPATH + "best_model.pt")

def train_salmon_detection_model(datapath, epochs, lr, device):
    
    # Create directory for model
    cwd = os.getcwd()
    MODELPATH = cwd + "/salmon_detection_models/"
    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)

    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
    
    dataset_training, dataset_validation, dataset_test = make_datasets(datapath, "salmon")
    
    # Random seed for reproducibility
    g = torch.manual_seed(0)

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
        batch_size=5,
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

    # Create salmon detection model: num_classes = background + salmon
    num_classes = 2
    model = get_detection_model(num_classes)

    # move model to current device
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.2
    )

    ### Training

    train_loss_list = []
    lr_step_sizes = []
    validation_losses = []

    best_validation_loss = float('inf')

    for epoch in range(epochs):
        
        # initialize tqdm progress bar
        prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
        
        train_loss_per_epoch = []

        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_per_epoch.append(loss_value)
            losses.backward()
            optimizer.step()

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
        validation_loss /= len(data_loader_validation)
        validation_losses.append(validation_loss)
        
        # Save the model if it has the best validation loss
        if validation_loss < best_validation_loss and epoch > 10:
            best_validation_loss = validation_loss
            best_model_path = os.path.join(MODELPATH, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
        
        # Change step size in optimizer
        lr_scheduler.step()

    ### SAVING RESULTS
    dict = {'training_loss': train_loss_list,  'validation_loss': validation_losses, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(MODELPATH + 'metrics.csv', index=False)

def test_detection_model(datapath, modelpath, device, task):
    
    if task == "salmon":
        num_classes = 2
        metrics_path = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/salmondetection/test_metrics.txt"
    if task == "bodyparts":
        num_classes = 6
        metrics_path = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/bodypartdetection/test_metrics.txt"
    
    model = get_detection_model(num_classes)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.eval().to(device)
    
    _, _, dataset_test = make_datasets(datapath, task)
    
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
    
    metrics = MeanAveragePrecision(iou_type="bbox")

    for i, data in enumerate(prog_bar):
        
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move target tensors to the same device
        
        with torch.no_grad():
            preds = model(images)
        
        # Visualize model prediction
        #utils.visualize_preds(task, images[0], preds[0], targets[0])
        
        metrics.update(preds, targets)
    
    metrics = metrics.compute()
    pprint(metrics)
    
    with open(metrics_path, "w") as f:
        pprint(metrics, stream=f)