import torch
from evaluate import calc_TPR_and_FPR
from torchvision import transforms
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from feature_extraction import collate_fn, get_resnet101_noclslayer, train_SVM, test_SVM, predict_features, create_results_folder, save_summary, analyze_data_tsne, explain_extractor, train_closedset, test_closedset, get_resnet101_withclslayer, plot_explained_individuals
from torchvision.models import ResNet101_Weights
from ReidentificationDataset import ReidentificationDataset
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, accuracy_score, auc
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay

def triplet_training_parameters_old():
    ''''epochs': 5,
        'optimizer': {
                'type': 'sgd',
                'lr': 0.001,
                'momentum': 0.9},
        'criterion': {
            'type': 'triplet_loss',
            'margin': 0.2,
            'p': 2,
            'eps': 1e-7
        },
        'batch_size': 2,
        'lr_scheduler': {
            'type': 'step_lr',
            'step_size': 20,
            'gamma': 0.1
        },
        'data_augmentation': {
            'color_jitter': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.4},
            'random_resized_crop':{
                'size': (224, 224), 
                'scale': (0.8, 1.0),
                'ratio': (0.9, 1.1)
            }},'''
            
    '''    LR = hyperparameters['optimizer']['lr']
        MOM = hyperparameters['optimizer']['momentum']
        STEP = hyperparameters['lr_scheduler']['step_size']
        GAMMA = hyperparameters['lr_scheduler']['gamma']
        MARG = hyperparameters['criterion']['margin']
        P = hyperparameters['criterion']['p']
        EPS = hyperparameters['criterion']['eps']'''
        
    '''    model.train()
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOM)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    # Loss criterion
    criterion = TripletMarginLoss(margin=MARG, p=P, eps=EPS)
    
    # TRAIN FEATURE EXTRACTOR
    
    # Get hyperparameters
    EPOCHS = hyperparameters['epochs']
    BS = hyperparameters['batch_size']
    NWORKERS = 5 if torch.cuda.is_available() else 0'''
    
    # Random seed for reproducibility
    #g = torch.manual_seed(0)
    
    '''    train_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BS,
        shuffle=True,
        num_workers=NWORKERS,
        collate_fn=collate_fn,
        generator=g
    )'''
    
    '''validation_loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=BS,
        shuffle=True,
        num_workers=NWORKERS,
        collate_fn=collate_fn,
        generator=g
    )'''
    
    #modelpath = train_extractor(model, criterion, optimizer, scheduler, train_loader, validation_loader, EPOCHS, hyperparameters, device, folder)

def annotate_axes(ax, text, c, fontsize=8):
    ax.text(0.85, 0.15, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=fontsize, color=c)
    
def give_name(target):
    names = {3:'Novak', 5:'Jannik', 7:'Casper', 9:'Holger', 10:'Roger', 17:'Alexander', 19:'Stefanos', 20:'Daniil'}
    return names[target]

def visualize_dataloader(bodypart, num_cols, dataloader):
    
    num_images = len(dataloader.sampler)
    
    iter_loader = iter(dataloader)
    
    images = []
    targets = []

    image, target = 0, 0
    for i in range(len(dataloader.sampler)):
        image, target = next(iter_loader)
        image = image[0]
        target = target[0].item()
        
        image = np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0))
        
        images.append(image)
        targets.append(give_name(target))
        
    
    
    # Plot the images in a grid
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    
    axes = axes.flatten()
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:  # Only plot if there are images left
            ax.imshow(images[i], aspect="auto")
            ax.axis('off')
            annotate_axes(ax, str(targets[i][0]), 'yellow')
        else:  # Hide the extra subplots
            ax.axis('off')
            ax.set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    fig.savefig('tailfin_dataset_test.png')     
              
def dataloaders(bodypart, hyperparameters, approach):
    
    modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/feature_extraction_models/newmodel/featuremodel_2024-04-25_14-23-54/best_model.pt"
    datapath_train_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Train"
    datapath_validation_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Validation"
    datapath_test_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Test"
    datapath_train_idun = "/cluster/home/magnuwii/Laks_kroppsdeler_Identifikasjonssett/Train"
    datapath_validation_idun = "/cluster/home/magnuwii/Laks_kroppsdeler_Identifikasjonssett/Validation"
    datapath_test_idun = "/cluster/home/magnuwii/Laks_kroppsdeler_Identifikasjonssett/Test"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datapath_train = datapath_train_idun if torch.cuda.is_available() else datapath_train_mac
    datapath_validation = datapath_validation_idun if torch.cuda.is_available() else datapath_validation_mac
    datapath_test = datapath_test_idun if torch.cuda.is_available() else datapath_test_mac
    
    NUMWORKERS = 10 if torch.cuda.is_available() else 0
    
    # Extract values from the dictionary
    BRIGHTNESS = hyperparameters['data_augmentation']['color_jitter']['brightness']
    CONTRAST = hyperparameters['data_augmentation']['color_jitter']['contrast']
    SATURATION = hyperparameters['data_augmentation']['color_jitter']['saturation']
    HUE = hyperparameters['data_augmentation']['color_jitter']['hue']
    SIZE = hyperparameters['data_augmentation']['random_resized_crop']['size']
    SCALE = hyperparameters['data_augmentation']['random_resized_crop']['scale']
    RATIO = hyperparameters['data_augmentation']['random_resized_crop']['ratio']
    BATCHSIZE = hyperparameters['batch_size']
    
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=SIZE,
                            scale=SCALE,
                            ratio=RATIO),
            transforms.ToTensor()
        ])
    validation_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=SIZE,
                            scale=SCALE,
                            ratio=RATIO),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=SIZE,
                            scale=SCALE,
                            ratio=RATIO),
            transforms.ToTensor()
        ])
    
    if approach == 2:
        
        # Data augmentation
        train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=BRIGHTNESS,
                            contrast=CONTRAST,
                            saturation=SATURATION,
                            hue=HUE),
        transforms.RandomResizedCrop(size=SIZE,
                                    scale=SCALE,
                                    ratio=RATIO),
        transforms.ToTensor()
        ])
        validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=BRIGHTNESS,
                            contrast=CONTRAST,
                            saturation=SATURATION,
                            hue=HUE),
        transforms.RandomResizedCrop(size=SIZE,
                                    scale=SCALE,
                                    ratio=RATIO),
        transforms.ToTensor()
        ])

    dataset_train = ReidentificationDataset(datapath_train, bodypart, train_transform)
    dataset_validation = ReidentificationDataset(datapath_validation, bodypart, validation_transform)
    dataset_test = ReidentificationDataset(datapath_test, bodypart, test_transform)
    
    g = torch.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=NUMWORKERS,
        collate_fn=collate_fn,
        generator=g
    )
    
    validation_loader = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=BATCHSIZE,
        shuffle=True,
        num_workers=NUMWORKERS,
        collate_fn=collate_fn,
        generator=g
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=NUMWORKERS,
        collate_fn=collate_fn,
        generator=g
    )
    
    return train_loader, validation_loader, test_loader, device

def results_folder():
    cwd = os.getcwd()
    MODELPATH = cwd + "/results/"
    folder = create_results_folder(MODELPATH)
    
    return folder

def analyze_features(df_train, df_test):
    
    result_df = pd.concat([df_train, df_test])

    # Reset the index if needed
    result_df.reset_index(drop=True, inplace=True)
    
    analyze_data_tsne(result_df)
    
def train_model(model, train_loader, validation_loader, hyperparameters, device, folder):
    
    EPOCHS = hyperparameters['epochs']
    LR = hyperparameters['optimizer']['lr']
    MOM = hyperparameters['optimizer']['momentum']
    STEP = hyperparameters['scheduler']['step_size']
    GAMMA = hyperparameters['scheduler']['gamma']
        
    criterion = CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOM)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    train_closedset(model, criterion, optimizer, scheduler, train_loader, validation_loader, EPOCHS, hyperparameters, device, folder)

def roc_models(bodyparts, hyperparameters, modelpaths, APPROACH):
    accuracies = []

    plt.figure(figsize=(8, 6))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Colors for bars, you can add more if needed
    
    
    for i, bp in enumerate(bodyparts):
        hyperparameters['bodypart'] = bp
        BODYPART = hyperparameters['bodypart']
        
        train_loader, validation_loader, test_loader, device = dataloaders(BODYPART, hyperparameters, APPROACH)
    
        print('device:', device)
        
        model_withcls = get_resnet101_withclslayer(ResNet101_Weights)
        model_withcls.to(device)
        state_dict = torch.load(modelpaths[i], map_location=torch.device('cpu'))
        model_withcls.load_state_dict(state_dict)
        model_nocls = get_resnet101_noclslayer(ResNet101_Weights)
        
        #best_model_path = train_model(model_withcls, train_loader, validation_loader, hyperparameters, device, folder)
        
        predictions, targets, probabilities, corrects = test_closedset(model_withcls, test_loader, device)
        
        accuracy = accuracy_score(predictions, targets)
        accuracies.append(accuracy)
        
        y_test = np.array(corrects)
        scores = np.array(probabilities)

        # Binarize the output
        map_individuals = {3: 0, 5: 1, 7: 2, 9: 3, 10: 4, 17: 5, 19: 6, 20: 7}
        mapped_y_test = [map_individuals[label] for label in y_test]
        y_train = train_loader.sampler.data_source.targets
        mapped_y_train = [map_individuals[label] for label in y_train]
        label_binarizer = LabelBinarizer().fit(mapped_y_train)
        y_onehot_test = label_binarizer.transform(mapped_y_test)
        

        fpr, tpr, _ = roc_curve(y_onehot_test.ravel(), scores.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'ROC curve of {bp} (area = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_accuracies(bodyparts, hyperparameters, modelpaths, APPROACH):
    
    # Determine the number of required subplots
    num_plots = len(bodyparts)

    # Calculate the number of rows and columns required
    num_rows = (num_plots - 1) // 3 + 1
    num_cols = min(num_plots, 3)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))
    
    # Flatten the axes array if it's more than 1D
    if num_plots == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    width = 0.5  # Width of the bars
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Colors for bars, you can add more if needed
    
    x = np.arange(8)  # Adjusting x positions for bars
    
    for i, bp in enumerate(bodyparts):
        
        train_loader, validation_loader, test_loader, device = dataloaders(bp, hyperparameters, APPROACH)
    
        print('device:', device)
        
        model_withcls = get_resnet101_withclslayer(ResNet101_Weights)
        model_withcls.to(device)
        state_dict = torch.load(modelpaths[i], map_location=torch.device('cpu'))
        model_withcls.load_state_dict(state_dict)
        
        #best_model_path = train_model(model_withcls, train_loader, validation_loader, hyperparameters, device, folder)
        
        predictions, targets, probabilities, corrects = test_closedset(model_withcls, test_loader, device, bp)

        correct_counts = dict()
        total_counts = dict()
        
        # Iterate through predictions and targets
        for pred, target in zip(predictions, targets):
            if not give_name(target) in total_counts:
                total_counts[give_name(target)] = 0
            total_counts[give_name(target)] += 1
            
            if pred == target:
                if not give_name(target) in correct_counts:
                    correct_counts[give_name(target)] = 0
                correct_counts[give_name(target)] += 1
        
        # Compute accuracy per class
        accuracy_per_class = {cls: correct_counts[cls] / total_counts[cls] for cls in total_counts}
        total_accuracy = sum(accuracy_per_class.values())/len(accuracy_per_class)
        
        axes[i].bar(x, accuracy_per_class.values(), width, label=bp, color=colors[i])
        axes[i].bar(8, total_accuracy, width, color='grey')
        axes[i].text(8, total_accuracy + 0.05, f'{total_accuracy:.2f}', ha='center')
        axes[i].set_ylabel('Accuracy')
        axes[i].set_ylim(0, 1.1)
        axes[i].axhline(y=1, color='gray', linestyle='--')
        #axes[i].set_title('Accuracy by Fish Species and Body Part')
        axes[i].set_xticks(np.arange(9))
        labels = list(accuracy_per_class.keys())
        labels.append('Avg. Accuracy')
        axes[i].set_xticklabels(labels, rotation=45, ha='right')  # Rotating labels for better readability
        axes[i].legend()
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.delaxes(axes[num_plots])
        
        
    
    print('yo')

    # Hide the empty plots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()

    plt.show()

def main():
    
    EPOCHS = 50 if torch.cuda.is_available() else 50
    BATCHSIZE = 5 if torch.cuda.is_available() else 1
    
    hyperparameters_AP1 = {
        'bodypart': 'thorax',
        'feature_extraction_model':{
            'type': 'ResNet101_noclslayer',
            'training': 'imagenet1k'},
        'classification': {
            'type': 'SVM',
            'kernel': 'linear',
            'c': 1.0,
            'randomstate': 42
        }
    }
    
    hyperparameters_AP2 = {
        'bodypart': 'tailfin',
        'epochs': EPOCHS,
        'data_augmentation': {
            'color_jitter': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.4},
            'random_resized_crop':{
                'size': (224, 224), 
                'scale': (0.8, 1.0),
                'ratio': (0.9, 1.1)
            }},
        'batch_size': BATCHSIZE,
        'classification_model':{
            'type': 'ResNet101',
            'training': 'imagenet1k+finetune'},
        'optimizer': {
            'type': 'sgd',
            'lr': 0.005,
            'momentum': 0.9
        },
        'scheduler': {
            'type': 'stepLR',
            'step_size': 10,
            'gamma': 0.2
        }
    }
    
    # CHOOSE APPROACH
    APPROACH = 2
    global hyperparameters
    if APPROACH == 1:
        hyperparameters = hyperparameters_AP1
    if APPROACH == 2:
        hyperparameters = hyperparameters_AP2
    
    
    
    #folder = results_folder()
    
    bodyparts = ['thorax', 'dorsalfin', 'pectoralfin', 'eyeregion', 'tailfin']
    
    modelpaths = ["/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-13-25/best_model.pt",
                  "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-45-11/best_model.pt",
                  "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_14-05-09/best_model.pt",
                  "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-38-49/best_model.pt",
                  "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_14-15-06/best_model.pt"]

    accuracies = []
    
    tot_targets = [[],[],[],[],[]]
    tot_predictions = [[],[],[],[],[]]
    bodyparts_titles = ['Thorax', 'Dorsal fin', 'Pectoral fin', 'Eye', 'Caudal fin']


    for i in range(5):
        hyperparameters_AP2['bodypart'] = bodyparts[i]
        BODYPART = hyperparameters['bodypart']
        MODELPATH = modelpaths[i]
        
        train_loader, validation_loader, test_loader, device = dataloaders(BODYPART, hyperparameters, APPROACH)

        print('device:', device)
        
        model_withcls = get_resnet101_withclslayer(ResNet101_Weights)
        model_withcls.to(device)
        state_dict = torch.load(MODELPATH, map_location=torch.device('cpu'))
        model_withcls.load_state_dict(state_dict)
        model_nocls = get_resnet101_noclslayer(ResNet101_Weights)
        
        #best_model_path = train_model(model_withcls, train_loader, validation_loader, hyperparameters, device, folder)
        
        if bodyparts_titles[i] == 'Thorax':
            predictions, targets, probabilities, corrects = test_closedset(model_withcls, test_loader, device, bodyparts_titles[i])
            
            tot_predictions[i] = predictions
            tot_targets[i] = targets
        
    try:
        with open('predictions_targets_step3_test.json', 'w') as f:
            json.dump({'predictions': tot_predictions, 'targets': tot_targets}, f)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
        
        #accuracy = accuracy_score(predictions, targets)
        #accuracies.append(accuracy)
        
        #plot_accuracies(bodyparts, hyperparameters, modelpaths, APPROACH)
        #roc_models(bodyparts, hyperparameters, modelpaths, APPROACH)
            
    print("yo")
    
    #df_train, df_test = predict_features(model, train_loader, test_loader, BODYPART, device)
    
    #analyze_features(df_train, df_test)
    
    #visualize_dataloader('thorax', 10, test_loader)
    
    #plot_explained_individuals(model_withcls, test_loader)
    
    #explain_extractor(model_withcls, 19, train_loader, test_loader)

    #classifier = train_SVM(hyperparameters, df_train)
    
    # TESTING CLASSIFIER
    
    #accuracy, report = test_SVM(classifier, df_test)
    
    '''    print('Bodypart:', BODYPART)
    print('Accuracy:', accuracy)
    print('Report:', report)
    
    save_summary(folder, hyperparameters, accuracy, report, train_loader, test_loader)'''
    

if __name__ == "__main__":
    main()