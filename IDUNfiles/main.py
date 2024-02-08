import torch
import argparse
from train_detector import train as train_detector

global device

def main():
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("GPU is available.")
    else:
        device = torch.device('cpu')
        print("GPU not available, running on CPU.")
        
    parser = argparse.ArgumentParser(description="Training FasterRCNN")
    
    parser.add_argument('--lr', type=float, default=0.005, help='Specify initial learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=100, help='Specify number of training epochs.')
    parser.add_argument('--datapath', type=str, default="/cluster/home/magnuwii/Helfisk_Deteksjonssett/", help='Path til datasett som skal brukes i trening.')
    args = parser.parse_args()
    
    lr = args.lr
    epochs = args.epochs
    datapath = args.datapath
    
    train_detector(datapath, epochs, lr, device)

if __name__ == "__main__":
    main()
    print("Script ending...")
