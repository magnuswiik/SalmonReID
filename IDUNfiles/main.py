import torch
import argparse
from train_detector import train as train_detector
from train_detector import test as test_detector
from inference_detector import inference as detector_inference

global device

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device: {device}")
        
    parser = argparse.ArgumentParser(description="Training FasterRCNN")
    
    parser.add_argument('--lr', type=float, default=0.005, help='Specify initial learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Specify number of training epochs.')
    parser.add_argument('--datapath', type=str, default="/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/", help='Path til datasett som skal brukes i trening.')
    parser.add_argument('--eval', type=bool, default=False, help="True if you are evaluating the model on its test set.")
    args = parser.parse_args()
    
    lr = args.lr
    epochs = args.epochs
    datapath = args.datapath
    eval = args.eval
    
    # DATAPATH FOR DETECTION WHOLE FISH
    # Mac: /Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/
    # IDUN: /cluster/home/magnuwii/Helfisk_Deteksjonssett/
    
    if not eval:
        train_detector(datapath, epochs, lr, device)
        
    else:
        #test_detector('/Users/magnuswiik/Documents/NTNU/5.klasse/Masteroppgave/masterthesis/IDUNfiles/models/model1/model1.pt', datapath, device)
        detector_inference("/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett/Images/", device)

if __name__ == "__main__":
    main()
    print("Script ending...")
