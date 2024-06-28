import torch.utils
import torchvision
import fasterrcnn.SalmonDataset as SalmonDataset
import fasterrcnn.LandmarksDataset as LandmarksDataset
import feature_extraction.ReidentificationDataset as ReidentificationDataset
import torch
from torchvision.models import ResNet101_Weights
from tqdm.auto import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.resnet import resnet101
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
import json
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def get_resnet101_withclslayer(weights, num_classes=8, modelpath=None):
    # Load the ResNet-101 model with pre-trained weights if specified
    model = resnet101(pretrained=weights)
    
    # Get the number of input features for the classification layer
    num_features = model.fc.in_features
    
    # Replace the classification layer with a new one
    model.fc = torch.nn.Linear(num_features, num_classes)
    
    return model

def get_detection_model(num_classes, weights=FasterRCNN_ResNet50_FPN_Weights):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def give_name(target):
    names = {3:'Novak', 5:'Jannik', 7:'Casper', 9:'Holger', 10:'Roger', 17:'Alexander', 19:'Stefanos', 20:'Daniil'}
    return names[target]

def plot_accuracies(bodyparts, predictions, targets):
    
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
    color3 = colors[0]
    color4 = colors[2]
    color5 = colors[3]
    color1 = colors[4]
    colors[2] = color3
    colors[3] = color4
    colors[0] = color1
    colors[4] = color5
    
    
    x = np.arange(8)  # Adjusting x positions for bars
    
    for i, bp in enumerate(bodyparts):
        correct_counts = {
            'Holger': 0,
            'Stefanos': 0,
            'Jannik': 0,
            'Roger': 0,
            'Casper': 0,
            'Novak': 0,
            'Daniil': 0,
            'Alexander': 0
        }
        
        total_counts = {
            'Holger': 0,
            'Stefanos': 0,
            'Jannik': 0,
            'Roger': 0,
            'Casper': 0,
            'Novak': 0,
            'Daniil': 0,
            'Alexander': 0
        }
        
        # Iterate through predictions and targets
        for pred, target in zip(predictions[i], targets[i]):
            target_name = give_name(target)
            pred_name = give_name(pred)
            total_counts[target_name] += 1
            if pred == target:
                correct_counts[pred_name] += 1
            
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
    plt.show()

@profile
def main():
    path1 = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_Deteksjonssett/"
    #path2 = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Deteksjonssett/"
    path3 = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/"
    datapath_train_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Train"
    datapath_validation_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Validation"
    datapath_test_mac = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Identifikasjonssett/Test"

    dataset = SalmonDataset.SalmonDataset(path1)
    
    g = torch.manual_seed(0)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g)

    modelpath1 = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/salmondetection/best_model.pt"
    modelpath2 = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/bodypartdetection/best_model.pt"
    modelpath3_thorax = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-13-25/best_model.pt"
    modelpath3_dorsalfin = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-45-11/best_model.pt"
    modelpath3_eye = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_13-38-49/best_model.pt"
    modelpath3_pectoralfin = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_14-05-09/best_model.pt"
    modelpath3_tailfin = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/featuremodel_2024-05-03_14-15-06/best_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes1 = 2
    num_classes2 = 6
    num_classes3 = 8
    
    model1 = get_detection_model(num_classes1)
    model1.load_state_dict(torch.load(modelpath1, map_location=torch.device('cpu')))
    model1.to(device)
    model1.eval()
    
    model2 = get_detection_model(num_classes2)
    model2.load_state_dict(torch.load(modelpath2, map_location=torch.device('cpu')))
    model2.to(device)
    model2.eval()
    
    model3_thorax = get_resnet101_withclslayer(ResNet101_Weights, num_classes3)
    model3_thorax.load_state_dict(torch.load(modelpath3_thorax, map_location=torch.device('cpu')))
    model3_dorsalfin = get_resnet101_withclslayer(ResNet101_Weights, num_classes3)
    model3_dorsalfin.load_state_dict(torch.load(modelpath3_dorsalfin, map_location=torch.device('cpu')))
    model3_eye = get_resnet101_withclslayer(ResNet101_Weights, num_classes3)
    model3_eye.load_state_dict(torch.load(modelpath3_eye, map_location=torch.device('cpu')))
    model3_pectoralfin = get_resnet101_withclslayer(ResNet101_Weights, num_classes3)
    model3_pectoralfin.load_state_dict(torch.load(modelpath3_pectoralfin, map_location=torch.device('cpu')))
    model3_tailfin = get_resnet101_withclslayer(ResNet101_Weights, num_classes3)
    model3_tailfin.load_state_dict(torch.load(modelpath3_tailfin, map_location=torch.device('cpu')))
    models = [model3_tailfin, model3_dorsalfin, model3_thorax, model3_pectoralfin, model3_eye]
    
    prog_bar = tqdm(dataloader, total=len(dataloader))
    
    map_individuals = {0:3, 1:5, 2:7, 3:9, 4:10, 5:17, 6:19, 7:20}
    individuals = [3, 5, 7, 9, 10, 17, 19, 20]
    bodyparts = ['Caudal fin','Dorsal fin', 'Thorax', 'Pectoral fin', 'Eye']
    
    reid_targets = [[],[],[],[],[]]
    reid_predictions = [[],[],[],[],[]]
    
    ### STEP 1

    for i, data in enumerate(prog_bar):
        
        images, targets = data
        
        individual = targets[0]['individual'].tolist()[0]
        
        if individual in individuals:
        
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                preds = model1(images)
            
            image1in = images[0]
            pred_scores = preds[0]['scores'].detach().cpu().tolist()
            pred_boxes = preds[0]['boxes'].detach().cpu().tolist()
            
            index_highconf = pred_scores.index(max(pred_scores))
            most_conf_predbox = pred_boxes[index_highconf]
            x1,y1,x2,y2 = most_conf_predbox
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            #image = image.detach().cpu().tolist()
            image1out = image1in.cpu().detach().numpy().transpose((1, 2, 0))
            cropped_image = image1out[y1:y2, x1:x2, :]
            
            # Show prediction
            '''fig, ax = plt.subplots()
            ax.imshow(cropped_image)
            ax.axis('off')
            plt.show()'''

            ### STEP 2
            
            image2in, individual = cropped_image, individual
            image2in = np.transpose(image2in, (2, 0, 1))
            image2in = torch.tensor(image2in)
            image2in = [image2in.to(device)]
            
            bodypart_confs = [0]*5
            
            bodyparts_pred = [0]*5
            
            with torch.no_grad():
                preds = model2(image2in)
                
            pred_scores = preds[0]['scores'].detach().cpu().tolist()
            pred_boxes = preds[0]['boxes'].detach().cpu().tolist()
            pred_labels = preds[0]['labels'].detach().cpu().tolist()
            image2out = image2in[0].cpu().detach().numpy().transpose((1, 2, 0))
            
            '''fig, ax = plt.subplots()
            ax.imshow(image2out)
            ax.axis('off')
            plt.show()'''
            
            for score, box, label in zip(pred_scores, pred_boxes, pred_labels):
                
                # HER STYRER DU HVILKEN BODY PART SOM SKAL LAGRES
                if label == 5:
                    index = label-1
                    if score > bodypart_confs[index]:
                        bodypart_confs[index] = score
                        bodyparts_pred[index] = box
            
            for i, box in enumerate(bodyparts_pred):
                
                '''fig, ax = plt.subplots()
                ax.imshow(image)
                ax.axis('off')'''
                
                if box != 0:
                    x1,y1,x2,y2 = box
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    cropped_image = image2out[y1:y2, x1:x2, :]
                    
                    '''rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    
                    plt.show()'''
                    
                    ### STEP 3
                    
                    SIZE = (224, 224)
                    SCALE = (0.8, 1.0)
                    RATIO = (0.9, 1.1)
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomResizedCrop(size=SIZE,
                                        scale=SCALE,
                                        ratio=RATIO),
                        transforms.ToTensor(),
                        ])
                    
                    image3in, individual = cropped_image, individual
                    #image3in = np.transpose(image3in, (2, 0, 1))
                    image3in = (image3in * 255).astype(np.uint8)
                    image3in = transform(image3in)
                    image3in = image3in.unsqueeze(0)
                    image3in = image3in.to(device)
                    model3 = models[i]
                    model3.to(device)
                    model3.eval()
                    
                    with torch.no_grad():
                        preds = model3(image3in)
                        p, pred = torch.max(preds, 1)
                        pred = pred.detach().item()
                        reid_predictions[i].append(map_individuals[pred])
                        reid_targets[i].append(individual)
                        
                    image3out = image3in[0].cpu().detach().numpy().transpose((1, 2, 0))
                    
                    '''if map_individuals[pred] != individual:
                        fig, ax = plt.subplots()
                        ax.imshow(image3out)
                        ax.axis('off')
                        ax.set_title(f'{bodyparts[i]}')
                        plt.figtext(0.5, 0.05, f'Predicted {give_name(map_individuals[pred])}, but {give_name(individual)} is the correct individual', wrap=True, horizontalalignment='center', fontsize=12)
                        plt.show()'''
    
    print('hiyeeeee')
    
    print("Predictions:", reid_predictions)
    print("Targets:", reid_targets)
    
    bodypart = "Eye"
    
    cmap=LinearSegmentedColormap.from_list('rg',["darkred", "w", "darkgreen"], N=256)
    target_names = ['Novak', 'Jannik', 'Casper', 'Holger', 'Roger', 'Alexander', 'Stefanos', 'Daniil']
    map_individuals = {0:3, 1:5, 2:7, 3:9, 4:10, 5:17, 6:19, 7:20}
    
    cm = confusion_matrix(reid_targets[4], reid_predictions[4])
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = np.trace(cm) / np.sum(cm)
    fig, ax = plt.subplots(figsize=(15,8))
    sns.heatmap(cmn, cmap=cmap, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)#, xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Test set results for Step 3. Body-part model: {bodypart}. Accuracy={round(accuracy, 3)}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'/Users/magnuswiik/Documents/masteroppgave figurer/pipelineresults/confmatrix_{bodypart}', dpi=600, transparent=True)
    plt.show()
                 
    
    plot_accuracies(bodyparts, reid_predictions, reid_targets)
    
    try:
        with open('predictions_targets.json', 'w') as f:
            json.dump({'predictions': reid_predictions, 'targets': reid_targets}, f)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
    

if __name__ == "__main__":
    main()