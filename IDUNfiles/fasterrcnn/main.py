import torch
import argparse
from train_detector import train as train_detector
from train_detector import test as test_detector
from train_detector import train_landmarks
from inference_detector import inference
from inference_detector import inference_landmark


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device: {device}")
        
    parser = argparse.ArgumentParser(description="Training FasterRCNN")
    
    parser.add_argument('--cloud', type=int, default=0, help="1 if running on IDUN. 0 if running on mac.")
    parser.add_argument('--task', type=str, default='salmon', help='salmon for salmon detection. landmarks for landmarks detection.')
    parser.add_argument('--lr', type=float, default=0.005, help='Specify initial learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Specify number of training epochs.')
    parser.add_argument('--train', type=int, default=1, help="1 if you are training the model.")
    args = parser.parse_args()
    
    cloud = args.cloud
    task = args.task
    lr = args.lr
    epochs = args.epochs
    train = args.train
    
    # DATAPATH FOR DETECTION WHOLE FISH
    # Mac: /Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/
    # IDUN: /cluster/home/magnuwii/Helfisk_Deteksjonssett/
    
    print("train:", train, "cloud:", cloud, "task:", task, "lr:", lr, "epochs:", epochs)
    
    if cloud:
        if train:
            if task == 'salmon':
                datapath = '/cluster/home/magnuwii/Helfisk_Deteksjonssett/'
                train_detector(datapath, epochs, lr, device)
            if task == 'landmarks':
                datapath = '/cluster/home/magnuwii/Helfisk_Landmark_Deteksjonssett_Trening/'
                train_landmarks(datapath, epochs, lr, device)
                
        else:
            datapath = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/Images/"
            modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/fasterrcnn/models/mobilemodel1/model1.pt"
            inference(datapath, modelpath, device)
    else:
        if train:
            if task == 'salmon':
                datapath = '/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/'
                train_detector(datapath, epochs, lr, device)
            if task == 'landmarks':
                datapath = '/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Landmark_Deteksjonssett_Trening/'
                train_landmarks(datapath, epochs, lr, device)
                
        else:
            '''datapath = '/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Landmark_Deteksjonssett/'
            modelpath = '/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/landmark_models/model1/model1.pt'
            inference_landmark(datapath, modelpath, device)'''
            
            datapath = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/"
            modelpath = "/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/fasterrcnn/models/model2/model2.pt"
            test_detector(datapath, modelpath, device)

if __name__ == "__main__":
    main()
    print("Script ending...")
