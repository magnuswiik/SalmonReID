import torch
import argparse
from detection import train_salmon_detection_model
from detection import test_detection_model
from detection import train_bodypart_detection_model

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device: {device}")
        
    parser = argparse.ArgumentParser(description="Training FasterRCNN")
    
    parser.add_argument('--cloud', type=int, default=0, help="1 if running on IDUN. 0 if running locally.")
    parser.add_argument('--task', type=str, default='salmon', help='salmon for salmon detection. bodyparts for bodypart detection.')
    parser.add_argument('--lr', type=float, default=0.005, help='Specify initial learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Specify number of training epochs.')
    parser.add_argument('--train', type=int, default=1, help="1 if you are training the model. 0 if you are testing the model.")
    args = parser.parse_args()
    
    cloud = args.cloud
    task = args.task
    lr = args.lr
    epochs = args.epochs
    train = args.train

    local_datapath_salmon = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_Deteksjonssett/"
    local_datapath_bodyparts = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Laks_kroppsdeler_Deteksjonssett/"
    cloud_datapath_salmon = "/cluster/home/magnuwii/Helfisk_Deteksjonssett/"
    cloud_datapath_bodyparts = "/cluster/home/magnuwii/Laks_kroppsdeler_Deteksjonssett/"
    
    print("train:", train, "cloud:", cloud, "task:", task, "lr:", lr, "epochs:", epochs)
    
    if cloud:
        if task == 'salmon':
            train_salmon_detection_model(cloud_datapath_salmon, epochs, lr, device)
        if task == 'bodyparts':
            train_bodypart_detection_model(cloud_datapath_bodyparts, epochs, lr, device)
    else:
        if train:
            if task == 'salmon':
                train_salmon_detection_model(local_datapath_salmon, epochs, lr, device)
            if task == 'bodyparts':
                train_bodypart_detection_model(local_datapath_bodyparts, epochs, lr, device)
                
        else:
            if task == "salmon":
                modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/salmondetection/best_model.pt"
                test_detection_model(local_datapath_salmon, modelpath, device, task)
            if task == "bodyparts":
                modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/results/IDUN/bodypartdetection/best_model.pt"
                test_detection_model(local_datapath_bodyparts, modelpath, device, task)

if __name__ == "__main__":
    main()
    print("Script ending...")
