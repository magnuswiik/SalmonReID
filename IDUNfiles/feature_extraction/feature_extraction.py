import torch
from torchvision.models.resnet import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from torch.nn import TripletMarginLoss
from torch import nn
import os, re, math
import pandas as pd
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from ReidentificationDataset import ReidentificationDataset
from TripletReidentificationDataset import TripletReidentificationDataset
from WholeFishReidentificationDataset import WholeFishReidentificationDataset
from ClosedSetReidentificationDataset import ClosedSetReidentificationDataset
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import captum
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
import torchvision.transforms as transforms
from PIL import Image
import random
from datetime import datetime
from sklearn.svm import SVC
from  matplotlib.colors import LinearSegmentedColormap

def create_results_folder(base_path, prefix="featuremodel"):
    # Create a timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Construct folder name with prefix and timestamp
    folder_name = f"{prefix}_{timestamp}"
    # Join with base path to create the full path
    folder_path = os.path.join(base_path, folder_name)
    # Create the folder
    os.makedirs(folder_path)
    return folder_path

def save_hyperparameters(folder_path, hyperparameters):
    # Define the file path for saving hyperparameters
    file_path = os.path.join(folder_path, "hyperparameters.txt")
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each hyperparameter to the file
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")
            
def save_summary(folder_path, hyperparameters, accuracy, report, train_loader, test_loader):
    # Define the file path for saving hyperparameters
    filepath = os.path.join(folder_path, "summary.txt")
    
    train_images = train_loader.sampler.data_source.images
    test_images = test_loader.sampler.data_source.images
    
    with open(filepath, 'w') as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")
        file.write(f'Dataset training indices: {train_images}\n')
        #file.write(f'Dataset validation indices: {val_indices}\n')
        file.write(f'Dataset test indices: {test_images}\n')
        file.write(f'Accuracy: {accuracy}\n')
        file.write('Classification Report:\n')
        file.write(report)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_resnet50_noclslayer(weights, modelpath=None):
    
    model = resnet50(weights)
    
    # Remove classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    return model

def get_resnet101_noclslayer(weights, modelpath=None):
    
    model = resnet101(weights)
    
    # Remove classification layer
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    
    return model

def get_resnet101_withclslayer(weights, num_classes=8, modelpath=None):
    # Load the ResNet-101 model with pre-trained weights if specified
    model = resnet101(pretrained=weights)
    
    # Get the number of input features for the classification layer
    num_features = model.fc.in_features
    
    # Replace the classification layer with a new one
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def get_resnet18_withclslayer(weights, num_classes=7, modelpath=None):
    # Load the ResNet-101 model with pre-trained weights if specified
    model = resnet18(pretrained=weights)
    
    # Get the number of input features for the classification layer
    num_features = model.fc.in_features
    
    # Replace the classification layer with a new one
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def make_datasets_with_ratio(datapath, landmark, hyperparameters):
    
    BRIGHT = hyperparameters['data_augmentation']['color_jitter']['brightness']
    CONTR = hyperparameters['data_augmentation']['color_jitter']['contrast']
    HUE = hyperparameters['data_augmentation']['color_jitter']['hue']
    SAT = hyperparameters['data_augmentation']['color_jitter']['saturation']
    
    SIZE = hyperparameters['data_augmentation']['random_resized_crop']['size']
    SCALE = hyperparameters['data_augmentation']['random_resized_crop']['scale']
    RATIO = hyperparameters['data_augmentation']['random_resized_crop']['ratio']
    
    # Random seed for reproducibility
    random.seed(0)
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=BRIGHT, contrast=CONTR, saturation=SAT, hue=HUE),
        transforms.RandomResizedCrop(size=SIZE, scale=SCALE, ratio=RATIO),
        transforms.ToTensor(),
    ])

    # use our dataset and defined transformations
    dataset = TripletReidentificationDataset(datapath, landmark, transform)
    dataset_validation = TripletReidentificationDataset(datapath, landmark, transform)
    dataset_test = TripletReidentificationDataset(datapath, landmark, transform)

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

def visualize_triplets(anchor, positive, negative):
    
    # Visualize the images
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(anchor.squeeze(0).cpu().permute(1, 2, 0))  # Convert back to CPU and permute dimensions
    axs[0].set_title('Anchor')
    axs[1].imshow(positive.squeeze(0).cpu().permute(1, 2, 0))
    axs[1].set_title('Positive')
    axs[2].imshow(negative.squeeze(0).cpu().permute(1, 2, 0))
    axs[2].set_title('Negative')
    plt.show()

def visualize_batch(image_batch):
    
    batch_dim = image_batch.size(0)
    
    # Visualize the images
    fig, axs = plt.subplots(1, batch_dim, figsize=(10, 3))

    for i in range(batch_dim):
        image = image_batch[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
        if batch_dim == 1:
            axs.imshow(image)
            axs.set_title(f'Image {i+1}')
        else:
            axs[i].imshow(image)
            axs[i].set_title(f'Image {i+1}')

    plt.show()

def train_extractor(model, criterion, optimizer, scheduler, train_loader, validation_loader, EPOCHS, hyperparameters, device, folder):
    
    best_model_path = os.path.join(folder, 'best_model.pt')
    
    save_hyperparameters(folder, hyperparameters)
    
    train_loss_list = []
    lr_step_sizes = []
    validation_loss_list = []

    best_validation_loss = float('inf')
    
    for epoch in range(EPOCHS):
        
        train_loss_per_epoch = 0
        validation_loss_per_epoch = 0
        
        
        prog_bar = tqdm(train_loader, total=len(train_loader.batch_sampler))
        
        for i, data in enumerate(prog_bar):
            images, targets = data
            optimizer.zero_grad()
            loss = 0
            with torch.set_grad_enabled(True):
                for triplet in images:
                    input = torch.stack(triplet)
                    output = model(input)
                    
                    loss += criterion(output[0], output[1], output[2])
                    #visualize_triplets(triplet[0], triplet[1], triplet[2])
                    
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()*len(images)*3
            train_loss_per_epoch += batch_loss
        
            prog_bar.set_description(desc=f"|Epoch: {epoch+1}/{EPOCHS}| Loss: {batch_loss:.4f}")
        
        for images, targets in validation_loader:
            val_loss = 0
            
            with torch.no_grad():
                for triplet in images:
                    input = torch.stack(triplet)
                    output = model(input)
                    
                    val_loss += criterion(output[0], output[1], output[2])
            
            batch_loss = val_loss.item()*len(images)*3
            validation_loss_per_epoch += batch_loss
        
        average_train_loss_this_epoch = train_loss_per_epoch/len(train_loader.sampler)
        train_loss_list.append(round(average_train_loss_this_epoch,5))
        average_validation_loss_this_epoch = validation_loss_per_epoch/len(validation_loader.sampler)
        validation_loss_list.append(round(average_validation_loss_this_epoch,5))
        lr_step_sizes.append(optimizer.param_groups[0]['lr'])
        
        
        # Save the model if it has the best validation loss
        if average_validation_loss_this_epoch < best_validation_loss:
            best_validation_loss = average_validation_loss_this_epoch
            torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()
        
    dict = {'training_loss': train_loss_list, 'validation_loss':validation_loss_list, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder, 'metrics.csv'), index=False)
    
    return best_model_path
    
def train_closedset(model, criterion, optimizer, scheduler, train_loader, validation_loader, EPOCHS, hyperparameters, device, folder):
    
    map_individuals = {3:0, 5:1, 7:2, 9:3, 10:4, 17:5, 19:6, 20:7}
    
    best_model_path = os.path.join(folder, 'best_model.pt')
    model.to(device)
    model.train()
    
    train_size = len(train_loader.sampler)
    validation_size = len(validation_loader.sampler)
    
    train_loss_list = []
    train_acc_list = []
    lr_step_sizes = []
    validation_loss_list = []
    validation_acc_list = []
    
    best_validation_loss = float('inf')
    
    for epoch in range(EPOCHS):
        
        prog_bar = tqdm(train_loader, total=len(train_loader))
        
        train_loss_per_epoch = 0
        validation_loss_per_epoch = 0
        train_running_corrects = 0
        validation_running_corrects = 0
        
        for i, data in enumerate(prog_bar):
            images, targets = data
            images = torch.stack(images).to(device)
            targets = torch.tensor([map_individuals[target.item()] for target in targets]).to(device)
            
            visualize_batch(images)

            optimizer.zero_grad()
            loss = 0
            with torch.set_grad_enabled(True):
                output = model(images)
                _, preds = torch.max(output, 1)
                loss = criterion(output, targets)
            
                loss.backward()
                optimizer.step()
            
            batch_loss = loss.item()*len(images)
            train_loss_per_epoch += batch_loss
            train_running_corrects += torch.sum(preds == targets.data)
                
            prog_bar.set_description(desc=f"|Epoch: {epoch+1}/{EPOCHS}| Loss: {batch_loss:.4f}")
        
        for images, targets in validation_loader:
            images = torch.stack(images).to(device)
            targets = torch.tensor([map_individuals[target.item()] for target in targets]).to(device)
            
            val_loss = 0
            with torch.no_grad():
                output = model(images)
                _, preds = torch.max(output, 1)
                val_loss += criterion(output, targets)
            
            batch_loss = val_loss.item()*len(images)
            validation_loss_per_epoch += batch_loss
            validation_running_corrects += torch.sum(preds == targets.data)
            
        train_epoch_acc = train_running_corrects / train_size
        train_acc_list.append(round(train_epoch_acc.item(), 5))
        validation_epoch_acc = validation_running_corrects / validation_size
        validation_acc_list.append(round(validation_epoch_acc.item(), 5))
        
        average_train_loss_this_epoch = train_loss_per_epoch/len(train_loader.sampler)
        train_loss_list.append(round(average_train_loss_this_epoch,5))
        average_validation_loss_this_epoch = validation_loss_per_epoch/len(validation_loader.sampler)
        validation_loss_list.append(round(average_validation_loss_this_epoch,5))
        lr_step_sizes.append(optimizer.param_groups[0]['lr'])
        
        
        # Save the model with best validation loss
        if average_validation_loss_this_epoch < best_validation_loss:
            best_validation_loss = average_validation_loss_this_epoch
            torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()
        print(f'Training Loss: {average_train_loss_this_epoch:.4f} Acc: {train_epoch_acc:.4f}') 
    
    
    dict = {'training_loss': train_loss_list, 'validation_loss':validation_loss_list, 'training_acc': train_acc_list, 'validation_acc':validation_acc_list, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder, 'metrics.csv'), index=False)
    
    return best_model_path

def give_name(target):
    names = {3:'Novak', 5:'Jannik', 7:'Casper', 9:'Holger', 10:'Roger', 17:'Alexander', 19:'Stefanos', 20:'Daniil'}
    return names[target]
     
def test_closedset(model, test_loader, device, bodypart):
    
    cmap=LinearSegmentedColormap.from_list('rg',["darkred", "w", "darkgreen"], N=256) 
    target_names = ['Novak', 'Jannik', 'Casper', 'Holger', 'Roger', 'Alexander', 'Stefanos', 'Daniil']
    map_individuals = {0:3, 1:5, 2:7, 3:9, 4:10, 5:17, 6:19, 7:20}
    map_individuals_reversed = {3:0, 5:1, 7:2, 9:3, 10:4, 17:5, 19:6, 20:7}
    
    model.to(device)
    model.eval()
    
    prog_bar = tqdm(test_loader, total=len(test_loader))
    
    probabilities_list = []
    correct_list = []
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        
        for i, data in enumerate(prog_bar):
            image, target = data
            
            image = torch.stack(image).to(device)
            target = target[0].to(device)

            #visualize_batch(image)

            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            p, pred = torch.max(output, 1)
            
            probabilities= probs.squeeze(0).tolist()
            probabilities_list.append(probabilities)
            correct_list.append(target.item())
            predictions_list.append(map_individuals[pred.item()])
            targets_list.append(target.item())
            
            pred_it = pred.item()
            target_it = target.item()
            pred_prob = round(probabilities[pred_it], 3)
            target_prob = round(probabilities[map_individuals_reversed[target_it]], 3)
                        
            if (give_name(map_individuals[pred_it]) != give_name(target_it)):# and ((give_name(map_individuals[pred_it]) == "Holger") or (give_name(map_individuals[pred_it]) == "Stefanos")):
                
                image_out = image[0].cpu().detach().numpy().transpose((1, 2, 0))
                
                fig, ax = plt.subplots()
                target_ex = torch.tensor(map_individuals_reversed[target.item()])
                explained_image = explain_image(model, image, target_ex)
                
                ax.imshow(explained_image)
                ax.axis('off')
                ax.set_title(f'{bodypart}')
                plt.figtext(0.5, 0.05, f'Predicted: {give_name(map_individuals[pred_it])}({pred_prob}). True: {give_name(target_it)}({target_prob})', horizontalalignment='center', fontsize=12)
                plt.savefig(f'/Users/magnuswiik/Documents/masteroppgave figurer/step3/predictions/Predicted({give_name(map_individuals[pred_it])})True({give_name(target_it)}{i})', dpi=600, transparent=True)
                        
    # Calculate accuracy
    accuracy = accuracy_score(targets_list, predictions_list)
    
    # Generate classification report
    report = classification_report(targets_list, predictions_list)
    
    '''cm = confusion_matrix(targets_list, predictions_list)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / np.sum(cm)
    fig, ax = plt.subplots(figsize=(15,8))
    sns.heatmap(cmn, cmap=cmap, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)#, xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Test set results for Step 3. Body-part model: {bodypart}. Accuracy={round(accuracy, 3)}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()'''
    
    return predictions_list, targets_list, probabilities_list, correct_list
    
def predict_features(model, train_loader, test_loader, landmark, device):
    
    labels_map = {'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5}
    
    landmark = labels_map[landmark]
    
    num_columns = 2050 # Resnet feature vector + fish id + landmark id

    df_train = pd.DataFrame(columns=range(num_columns))
    df_test = pd.DataFrame(columns=range(num_columns))
    
    model.eval()
    
    for image, target in train_loader:
        image = torch.stack(image)
        fish = target[0].item()
        
        output = model(image)
        output_flattened = output.view(1, -1).squeeze(0).detach().numpy()
        output_flattened = np.append(output_flattened, [fish, landmark])
        df_train.loc[len(df_train)] = output_flattened
        
    
    '''    for images, targets in validation_loader:
        
        image = images[0][0].unsqueeze(0)
        fish = targets[0][0]
        
        output = model(image)
        output_flattened = output.view(1, -1).squeeze(0).detach().numpy()
        output_flattened = np.append(output_flattened, [fish, landmark])
        df_train.loc[len(df_train)] = output_flattened'''
        
    for image, target in test_loader:
        image = torch.stack(image)
        fish = target[0].item()
        
        output = model(image)
        output_flattened = output.view(1, -1).squeeze(0).detach().numpy()
        output_flattened = np.append(output_flattened, [fish, landmark])
        df_test.loc[len(df_test)] = output_flattened
                        
    return df_train, df_test
    
def analyze_data_pca(df):
    
    labels_map = {1:'tailfin', 2:'dorsalfin', 3:'thorax', 4:'pectoralfin', 5:'eyeregion'}
    
    df = df[df['2049'] == 3]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, :-2])

    # Define the number of principal components you want to retain
    n_components = 10  # For example, retaining 10 principal components

    # Perform PCA
    pca = PCA(n_components=n_components)
    T = pca.fit_transform(scaled_data)
    P = pca.components_.T
    exp_var = pca.explained_variance_ratio_
    
    PCs = [0,1]
    
    scores = pd.DataFrame(data=T, columns=[f"PC{i+1}" for i in range(n_components)])
    
    scores["fish"] = df.iloc[:, -2].reset_index(drop=True)
    scores["landmark"] = df.iloc[:, -1].reset_index(drop=True)

    sns.set_theme()

    fig = plt.figure()
    
    unique_landmark_ids = scores["landmark"].unique()
    unique_fish_ids = scores["fish"].unique()
    for i, fish_id in enumerate(unique_fish_ids):
        fish_scores = scores[scores["fish"] == fish_id]

        plt.scatter(
            fish_scores["PC1"],
            fish_scores["PC2"],
            label=int(fish_id)
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Fish ID")
    plt.show()
    
def analyze_data_tsne(df):
    
    labels_map = {1:'tailfin', 2:'dorsalfin', 3:'thorax', 4:'pectoralfin', 5:'eyeregion'}
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=0)
    
    result = tsne.fit_transform(df)
    
    result = pd.DataFrame(data=result)
    result["fish"] = df.iloc[:, -2].reset_index(drop=True)
    result["landmark"] = df.iloc[:, -1].reset_index(drop=True)
    
    unique_fish_ids = result["fish"].unique()
    for i, fish_id in enumerate(unique_fish_ids):
        fish_result = result[result["fish"] == fish_id]

        plt.scatter(
            fish_result[0],
            fish_result[1],
            label=int(fish_id)
        )

    
    #plt.scatter(result[:, 0], result[:, 1])
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title="Fish ID")
    plt.show()
    
def train_SVM(hyperparameters, df_train):
    
    labels = df_train[2048]
    feature_vectors = df_train.drop(columns=[2048,2049])
    
    # Get hyperparameters for SVM
    KERNEL = hyperparameters['classification']['kernel']
    C = hyperparameters['classification']['c']
    RANDOMSTATE = hyperparameters['classification']['randomstate']
    classifier = SVC(kernel=KERNEL, C=C, random_state=RANDOMSTATE)
    
    # Train the SVM classifier
    classifier.fit(feature_vectors, labels)

    return classifier

def test_SVM(classifier, df_test):
    
    labels = df_test[2048]
    feature_vectors = df_test.drop(columns=[2048,2049])
    
    # Predict labels for the feature vectors
    predicted_labels = classifier.predict(feature_vectors)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    
    # Generate classification report
    report = classification_report(labels, predicted_labels)
    
    return accuracy, report
    
def explain_extractor_old(model, individual, train_loader, test_loader):
    
    map_individuals = {3:0, 5:1, 7:2, 9:3, 10:4, 17:5, 19:6, 20:7}
    
    iter_test_loader = iter(test_loader)

    image, target = 0, 0
    for i in range(len(test_loader.sampler)):
        image, target = next(iter_test_loader)
        image = image[0].unsqueeze(0)
        target = target[0].item()
        
        if target == individual:
            integrated_gradients = IntegratedGradients(model)
            attributions_ig = integrated_gradients.attribute(image, target=map_individuals[target], n_steps=50)
            
            image = np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0))
            
            '''    fig1, ax1 = viz.visualize_image_attr(None, image, method="original_image", title="Input image to ResNet")
            fig2, ax2 = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().detach().cpu().numpy(), (1,2,0)),
                                        image,
                                        method='heat_map',
                                        sign='positive',
                                        title='Integrated gradients')'''
            
            attributions_np = np.transpose(attributions_ig.squeeze().detach().cpu().numpy(), (1, 2, 0))

            # Set negative values to zero
            attributions_np[attributions_np < 0] = 0

            # Scale the attributions to the range [0, 1] for overlay
            attributions_np = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min())

            # Ensure that attributions_np is in the appropriate range [0, 1]
            attributions_np = np.clip(attributions_np, 0, 1)
            
            # Display the input image and the explained image
            plt.figure(figsize=(10, 5))

            plt.subplot(3, 1, 1)
            plt.imshow(image)
            plt.title("Input image")
            plt.axis('off')

            plt.subplot(3, 1, 2)
            plt.imshow(attributions_np, cmap='hot')
            plt.title("Attributions")
            plt.axis('off')
            
            attributions_np += 1
            
            # Increase brightness based on attributions
            brightness_adjusted_image = np.clip(image * attributions_np, 0, 1)

            plt.subplot(3, 1, 3)
            plt.imshow(brightness_adjusted_image)
            plt.title("Overlayed")
            plt.axis('off')

            plt.show()

    '''    # Save figures
    fig1.savefig('input_image.png')
    fig2.savefig('explained_image.png')'''

def annotate_axes(ax, text, c, fontsize=12):
    ax.text(0.85, 0.1, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, color=c)
    
def explain_image(model, image, target):
    
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(image, target=target, n_steps=50)
    
    image = np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0))
    attributions_np = np.transpose(attributions_ig.squeeze().detach().cpu().numpy(), (1, 2, 0))
    
    attributions_np[attributions_np < 0] = 0
    attributions_np = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min())
    attributions_np = np.clip(attributions_np*5, 0, 1)
    
    # Overlay the image with the attributions
    attr_np = attributions_np + 1
    overlayed_image = np.clip(image * attr_np, 0, 1)

    return overlayed_image

def plot_explained_individuals(model, data_loader):
    
    map_individuals = {3:0, 5:1, 7:2, 9:3, 10:4, 17:5, 19:6, 20:7}
    
    num_cols = 8
    num_rows = 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 1))
    
    individuals = []
    images = []
    
    iter_test_loader = iter(data_loader)
    
    for i in range(len(data_loader.sampler)):
        image, target = next(iter_test_loader)
        image = image[0].unsqueeze(0)
        target = target[0].item()
        
        if target not in individuals:
            
            explained_image = explain_image(model, image, map_individuals[target])
            
            images.append(explained_image)
            individuals.append(target)
    
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        axes[i].imshow(img, aspect=1)
        axes[i].axis('off')
        #annotate_axes(axes[i], str(i//3 + 1), 'yellow')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    fig.savefig('individuals_explained_eye.png')
     
def explain_extractor(model, individual, train_loader, test_loader):
    
    map_individuals = {3:0, 5:1, 7:2, 9:3, 10:4, 17:5, 19:6, 20:7}
    
    iter_test_loader = iter(test_loader)
    plot_images = []

    for i in range(len(test_loader.sampler)):
        image, target = next(iter_test_loader)
        image = image[0].unsqueeze(0)
        target = target[0].item()
        
        with torch.no_grad:
            pred = model(image)
        
        if target == individual:
            integrated_gradients = IntegratedGradients(model)
            attributions_ig = integrated_gradients.attribute(image, target=map_individuals[target], n_steps=50)
            
            image = np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0))
            
            attributions_np = np.transpose(attributions_ig.squeeze().detach().cpu().numpy(), (1, 2, 0))

            attributions_np[attributions_np < 0] = 0
            attributions_np = (attributions_np - attributions_np.min()) / (attributions_np.max() - attributions_np.min())
            attributions_np = np.clip(attributions_np*5, 0, 1)
            
            # Overlay the image with the attributions
            attr_np = attributions_np + 1
            overlayed_image = np.clip(image * attr_np, 0, 1)
            
            # Append images to list
            plot_images.append(image)
            plot_images.append(attributions_np)
            plot_images.append(overlayed_image)

    # Plot the images in a grid
    num_cols = 6
    num_rows = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    
    axes = axes.flatten()
    
    for i, img in enumerate(plot_images):
        axes[i].imshow(img, aspect="auto")
        axes[i].axis('off')
        annotate_axes(axes[i], str(i//3 + 1), 'yellow')

    plt.subplots_adjust(wspace=.01, hspace=.01)
    plt.show()
    
    fig.savefig('testset19_tailfin_explained.png')

      
#train_closedset(path, hyperparameters, device)

#features.to_csv("/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/feature_extraction/features_thorax_NOTtrained.csv")

#features = pd.read_csv("/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/feature_extraction/features_thorax_trained.csv")

#labels = features[2049]

#feature_vectors = features.drop(columns=[2048,2049])

#train_SVM(feature_vectors.values, labels.values)

#analyze_data_tsne(features)

#train_extractor(datapath, 100, 0.005, "cpu")

#explain_extractor(modelpath,"/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Identifikasjonssett/fish9/thorax/fish9_thorax_GP020101_00005889.jpg", "cpu")

    