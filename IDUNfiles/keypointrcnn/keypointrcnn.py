import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from SalmonKeypointDataset import SalmonKeypointDataset, Resize
import os, time
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

def getkeypointmodel(num_classes, num_keypoints):
    model = keypointrcnn_resnet50_fpn(pretrained=False,
                                        pretrained_backbone=True,
                                        num_keypoints=num_keypoints,
                                        num_classes = num_classes)
    
    # Modify the box predictor to predict only 1 bounding box
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    return model


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def visualize_keypoints(images, targets):
    for img, target in zip(images, targets):
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # Plot the image
        plt.imshow(img_np)

        # Extract keypoints from the target
        keypoints = target['keypoints'].detach().cpu().numpy()
        bboxes = target['boxes'].detach().cpu().numpy()
        scores = target['scores'].detach().cpu().numpy()

        # Plot each keypoint
        for instance in keypoints:
            for keypoint in instance:
                x, y, _ = keypoint  # Extract x, y coordinates and visibility
                plt.scatter(x, y, color='red', s=10)  # Plot keypoint as a red point
        c = 0
        for box in bboxes:
            x1, y1, x2, y2 = box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='blue', linewidth=1)
            # plot confidence score
            plt.text(x2 - 50, y1 + 15, scores[c], color='white', fontsize=8, bbox=dict(facecolor='blue', alpha=0.5))
            c += 1

        
        plt.axis('off')
        plt.show()

data_path1 = '/cluster/home/magnuwii/Helfisk_Landmark_Deteksjonssett_Trening/'
data_path2 = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Landmark_Deteksjonssett_Trening/"

classes = ['tailfin', 'dorsalfin', 'thorax', 'pectoralfin', 'eyeregion']
keypoints = ['tailtop', 'tailbot', 'dorsalback', 'dorsalfront', 'pectoral', 'eye', 'snouttop', 'snoutbot']

# Random seed for reproducibility
g = torch.manual_seed(0)

# Define your custom dataset and data loader
dataset = SalmonKeypointDataset(data_path1)
data_loader_training = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g
    )

# Set the number of classes for your dataset
num_classes = 1 + 1 # add for background
num_keypoints = len(keypoints)

# Load the pre-trained model
model = getkeypointmodel(num_classes, num_keypoints)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



# Test the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/keypointrcnn/keypoint_rcnn_model_noresize.pt"
model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
model.eval()

print(model)

num_epochs = 1

count = 0

for epoch in range(num_epochs):
    
    prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Visualize the images and targets 
        #visualize_keypoints(images, targets)

        # Forward pass
        start = time.time()
        with torch.no_grad():
            outputs = model(images)
        end = time.time()
        print("Prediction time: ", end - start)
        visualize_keypoints(images, outputs)




# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
#modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/keypoint_rcnn_model.pt"
#model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
model.train()

#print(model)

train_loss_list = []
lr_step_sizes = []
validation_losses = []

num_epochs = 100

count = 0

for epoch in range(num_epochs):
    
    prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
    
    train_loss_per_epoch = []
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        m = images[0].shape
        #tb = targets[0]['boxes'].shape
        #tl = targets[0]['labels'].shape
        tk = targets[0]['keypoints'].shape
        #m2 = images[1].shape
        #tb2 = targets[1]['boxes'].shape
        #tl2 = targets[1]['labels'].shape
        #tk2 = targets[1]['keypoints'].shape
        
        # Visualize the images and targets 
        # visualize_keypoints(images, targets)

        # Forward pass
        outputs = model(images, targets)
        losses = sum(loss for loss in outputs.values())
        train_loss_per_epoch.append(losses.detach().cpu().item())
        count += 1

        # Backward pass and optimization
        losses.backward()
        optimizer.step()

    # Save metrics per epoch
    lr_step_sizes.append(optimizer.param_groups[0]['lr'])
    train_loss_list.append(sum(loss for loss in train_loss_per_epoch)/len(train_loss_per_epoch))
        

    # Update the learning rate
    lr_scheduler.step()
    
    # Print the loss for each epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}')
    
MODELPATH1 = "/cluster/home/magnuwii/masterthesis/IDUNfiles/keypointrcnn/models/model1/"
MODELPATH2 = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/keypointmodels/model1/"

if not os.path.exists(MODELPATH1):
    os.mkdir(MODELPATH1)
    
dict = {'training_loss': train_loss_list, 'lr_step_size': lr_step_sizes}
df = pd.DataFrame(dict)
df.to_csv(MODELPATH1 + 'metrics.csv', index=False)

# Save the trained model
torch.save(model.state_dict(), 'keypoint_rcnn_model_noresize.pt')
