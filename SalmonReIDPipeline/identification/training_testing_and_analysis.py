import torch
from torchvision.models.resnet import resnet101
from torch import nn
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from captum.attr import IntegratedGradients
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

def create_results_folder(base_path, prefix="reid_model"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{prefix}_{timestamp}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path)
    
    return folder_path

def save_hyperparameters(folder_path, hyperparameters):
    file_path = os.path.join(folder_path, "hyperparameters.txt")
    with open(file_path, "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")
            
def save_summary(folder_path, hyperparameters, accuracy, report, train_loader, test_loader):
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

def give_name(target):
    names = {3:'Novak', 5:'Jannik', 7:'Casper', 9:'Holger', 10:'Roger', 17:'Alexander', 19:'Stefanos', 20:'Daniil'}
    return names[target]

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_resnet101_withclslayer(weights, num_classes=8, modelpath=None):

    model = resnet101(pretrained=weights)
    
    num_features = model.fc.in_features
    
    # Replace the classification layer with a new one
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def visualize_batch(image_batch):
    
    batch_dim = image_batch.size(0)
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
    
def train_reid_model(model, criterion, optimizer, scheduler, train_loader, validation_loader, EPOCHS, hyperparameters, device, folder):
    
    # This function performs model training for the body part re-identification models
    
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

def test_reid_model(model, test_loader, device, bodypart):
    
    # This function performs model testing for the body part re-identification models
    
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
    
    return predictions_list, targets_list, probabilities_list, correct_list
    
def analyze_data_pca(df):
    
    # This functions analyse features from the layer before the classification layer of a model.
    
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
    
    # This functions analyse features from the layer before the classification layer of a model.
    
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

def annotate_axes(ax, text, c, fontsize=12):
    ax.text(0.85, 0.1, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, color=c)
    
def explain_image(model, image, target):
    
    # This function explains an image´s pixel importance for the model´s prediction
    
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
