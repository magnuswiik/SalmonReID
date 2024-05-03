import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import os
from feature_extraction import make_datasets, collate_fn


# Training method

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, datapath, hyperparameters, criterion, optimizer, scheduler, num_epochs=5):
    
    # Get hyperparameters
    EPOCHS = hyperparameters['epochs']
    BS = hyperparameters['batch_size']
    LR = hyperparameters['optimizer']['lr']
    MOM = hyperparameters['optimizer']['momentum']
    WD = hyperparameters['optimizer']['momentum']
    FACTOR = hyperparameters['lr_scheduler']['factor']
    PATIENCE = hyperparameters['lr_scheduler']['patience']
    NWORKERS = 5 if torch.cuda.is_available() else 0
    
    dataset_training, dataset_validation = make_datasets(datapath, hyperparameters)
    
    train_size = len(dataset_training)
    
    # Random seed for reproducibility
    g = torch.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_training,
        batch_size=BS,
        shuffle=False,
        num_workers=NWORKERS,
        collate_fn=collate_fn,
        generator=g
    )

    best_model_params_path = 'model_weights1.pt'
    
    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            
            inputs = torch.stack(inputs)
            labels = torch.tensor(labels)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
       
        scheduler.step()

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects / train_size

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save best model weights
    torch.save(model.state_dict(), best_model_params_path)
    return model


def main():
    
    # Initialize model

    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_classes = 7
    model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 2 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
    datapath_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Identifikasjonssett/"
    
    hyperparameters = {
        'model': 'ResNet101',
        'epochs': 100,
        'optimizer': {
                'type': 'sgd',
                'lr': 0.0005,
                'momentum': 0.9,
                'weight_decay': 0.0005},
        'batch_size': 2,
        'lr_scheduler': {
            'type': 'reduce_lr_on_plateau',
            'factor': 0.2,
            'patience': 5
        },
        'data_augmentation': {
            'color_jitter': {
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'hue': 0},
            'random_resized_crop':{
                'size': (224, 224), 
                'scale': (0.8, 1.0),
                'ratio': (0.9, 1.1)
            }}  
    }
    
    model_ft = train_model(model_ft, datapath_mac, hyperparameters, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

if __name__ == "__main__":
    main()