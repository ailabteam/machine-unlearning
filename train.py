# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import os

from utils import get_cifar10_loaders

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = 20  # You can start with 10-20 epochs for good results
LR = 0.001
CLASS_TO_FORGET = 3 # Class 'cat' in CIFAR-10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = './models'

# --- Helper Functions for Training and Testing ---
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader, desc=f"Training Epoch")
    total_loss = 0
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def run_training_pipeline(model_name, train_loader, test_loader):
    print(f"\n--- Starting Training for: {model_name} ---")
    
    # Initialize model
    model = models.resnet18(weights=None, num_classes=10) # From scratch
    model = model.to(DEVICE)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        train(model, DEVICE, train_loader, optimizer, criterion)
        current_acc = test(model, DEVICE, test_loader, criterion)
        
        # Save the model with the best validation accuracy
        if current_acc > best_acc:
            best_acc = current_acc
            save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path} with accuracy: {best_acc:.2f}%")
            
    print(f"--- Finished Training for: {model_name} ---")


if __name__ == '__main__':
    # Create directory to save models if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Train the full model (M_full) on all 10 classes
    full_train_loader, full_test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    run_training_pipeline("model_full", full_train_loader, full_test_loader)

    # 2. Train the retrained model (M_retrain) on 9 classes
    # This will be our "ground truth" for unlearning
    retrain_train_loader, retrain_test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, class_to_forget=CLASS_TO_FORGET)
    run_training_pipeline("model_retrain", retrain_train_loader, retrain_test_loader)
    
    print("\n--- All baseline models have been trained! ---")
