import torch
import argparse
from train_detector import train as train_detector

global device

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"device: {device}")
        
    parser = argparse.ArgumentParser(description="Training FasterRCNN")
    
    parser.add_argument('--lr', type=float, default=0.005, help='Specify initial learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Specify number of training epochs.')
    parser.add_argument('--datapath', type=str, default="/cluster/home/magnuwii/Helfisk_Deteksjonssett/", help='Path til datasett som skal brukes i trening.')
    args = parser.parse_args()
    
    lr = args.lr
    epochs = args.epochs
    datapath = args.datapath
    
    # DATAPATH
    # Mac: /Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Deteksjonssett
    # IDUN: /cluster/home/magnuwii/Helfisk_Deteksjonssett/
    
    train_detector(datapath, epochs, lr, device)

if __name__ == "__main__":
    main()
    print("Script ending...")
