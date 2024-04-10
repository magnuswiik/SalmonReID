import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torch.nn import TripletMarginLoss
import os, re
import pandas as pd
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
from ReidentificationDataset import ReidentificationDataset
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import captum
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torchvision.transforms as transforms
from PIL import Image

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

def train_extractor(datapath, epochs, lr, device):
    
    dataset = ReidentificationDataset(datapath, "thorax")
    
    # Random seed for reproducibility
    g = torch.manual_seed(0)
    
    data_loader_training = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=g
    )
    
    model = get_resnet50_noclslayer(ResNet50_Weights)
    model.train()
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_loss_list = []
    lr_step_sizes = []
    
    for epoch in range(epochs):
        
        train_loss_per_epoch = []
        
        prog_bar = tqdm(data_loader_training, total=len(data_loader_training))
        
        for i, data in enumerate(prog_bar):
            imgs, targets = data
            optimizer.zero_grad()
            anchor_outputs = model(imgs[0][0])
            positive_outputs = model(imgs[0][1])
            negative_outputs = model(imgs[0][2])
            triplet_loss = TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
            loss = triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
            train_loss_per_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            prog_bar.set_description(desc=f"|Epoch: {epoch+1}/{epochs}| Loss: {loss:.4f}")
        
        train_loss_list.append(sum(train_loss_per_epoch)/len(train_loss_per_epoch))
        lr_step_sizes.append(optimizer.param_groups[0]['lr'])
        
    cwd = os.getcwd()
    MODELPATH = cwd + "/feature_extraction_models/model1/"

    if not os.path.exists(MODELPATH):
        os.makedirs(MODELPATH)
        
    dict = {'training_loss': train_loss_list, 'lr_step_size': lr_step_sizes}
    df = pd.DataFrame(dict)
    df.to_csv(MODELPATH + 'metrics.csv', index=False)

    torch.save(model.state_dict(), MODELPATH + "model1.pt")
    print("Model is saved at:" + MODELPATH + "model1.pt")
                        

def extract_features(modelpath, datapath, device):
    
    labels_map = {'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5}
    
    num_columns = 2050 # Resnet feature vector + fish id + landmark id

    df = pd.DataFrame(columns=range(num_columns))
    
    model = get_resnet50_noclslayer(ResNet50_Weights)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.eval()
    
    for fish in sorted(os.listdir(datapath)):
        if (not fish.startswith('.')):
            for landmark in sorted(os.listdir(os.path.join(datapath, fish))):
                if (not landmark.startswith('.')):
                    for file in sorted(os.listdir(os.path.join(datapath, fish, landmark))):
                        if (not file.startswith('.')):
                            image_path = os.path.join(datapath, fish, landmark, file)
                            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                            image = np.transpose(image / 255.0, (2, 0, 1)) # Pixel values need to be between 0 and 1 and transpose image 
                            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension and add to device

                            output = model(image_tensor)
                            output_flattened = output.view(1, -1).squeeze(0).detach().numpy()
                            fish_id = [int(x) for x in re.findall(r'\d+', file)][0]
                            landmark_id = labels_map[landmark]
                            output_flattened = np.append(output_flattened, [fish_id, landmark_id])
                            df.loc[len(df)] = output_flattened
                        
    return df
    
    
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
    
    df = df[df[2049] == 3]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    
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
    
def train_SVM(feature_vectors, labels):
    
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=0)
    
    # Train the SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    
def explain_extractor(modelpath, datapath, device):
    
    image = cv2.cvtColor(cv2.imread(datapath), cv2.COLOR_BGR2RGB)
    image_trans = np.transpose(image / 255.0, (2, 0, 1)) # Pixel values need to be between 0 and 1 and transpose image 
    image_tensor = torch.tensor(image_trans, dtype=torch.float32).unsqueeze(0).to(device) 
    
    
    model = get_resnet50_noclslayer(ResNet50_Weights)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.eval()
    
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(image_tensor, target=1, n_steps=200)
    
    img = np.transpose(image_tensor.squeeze().cpu().detach().numpy(), (1,2,0))
    
    fig1, ax1 = viz.visualize_image_attr(None, img, method="original_image", title="Input image to ResNet")
    fig2, ax2 = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().detach().cpu().numpy(), (1,2,0)),
                                 img,
                                 method='heat_map',
                                 sign='positive',
                                 title='Integrated gradients')
    
    

def main ():
    
    modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/feature_extraction_models/thoraxmodel/model1.pt"
    datapath = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Prediksjoner_Identifikasjonssett/"
    
    #features = extract_features(modelpath, datapath, "cpu")
    
    #features.to_csv("/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/feature_extraction/features_thorax_NOTtrained.csv")
    
    #features = pd.read_csv("/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/feature_extraction/features_thorax_trained.csv")
    
    #labels = features[2049]
    
    #feature_vectors = features.drop(columns=[2048,2049])
    
    #train_SVM(feature_vectors.values, labels.values)
    
    #analyze_data_tsne(features)
    
    #train_extractor(datapath, 100, 0.005, "cpu")
    
    explain_extractor(modelpath,"/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Identifikasjonssett/fish9/thorax/fish9_thorax_GP020101_00005889.jpg", "cpu")
    
    
if __name__ == "__main__":
    
    main()
    