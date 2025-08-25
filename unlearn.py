# unlearn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # <<<<<<<<<<<<<<<< LỖI ĐÃ SỬA Ở ĐÂY
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import copy

from utils import get_cifar10_loaders

# --- Configuration ---
BATCH_SIZE = 256
UNLEARN_EPOCHS = 5
LR = 0.0001
CLASS_TO_FORGET = 3 # Class 'cat'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = './models'
FULL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'model_full.pth')

# --- Helper to load a pre-trained model ---
def load_model(path, device):
    model = models.resnet18(weights=None, num_classes=10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# --- New Helper Function ---
def get_forget_loader(class_to_forget, batch_size):
    """
    Creates a DataLoader containing ONLY the data from the class to be forgotten.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    
    forget_indices = [i for i, target in enumerate(trainset.targets) if target == class_to_forget]
    forget_set = torch.utils.data.Subset(trainset, forget_indices)
    
    print(f"Data loader for forgotten class '{class_to_forget}' created. Size: {len(forget_set)}")
    return torch.utils.data.DataLoader(forget_set, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Unlearning Method 1: Fine-tuning (FT) ---
def unlearn_finetuning(model, device, unlearn_loader, epochs, lr):
    print("\n--- Starting Unlearning via Fine-tuning ---")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(unlearn_loader, desc=f"FT Epoch {epoch}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
    
    print("--- Fine-tuning Unlearning Complete ---")
    return model

# --- Unlearning Method 2: Negative Gradient Ascent (GA) ---
def unlearn_negative_gradient(model, device, forget_loader, epochs, lr):
    print("\n--- Starting Unlearning via Negative Gradient Ascent ---")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(forget_loader, desc=f"GA Epoch {epoch}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = -1 * criterion(output, target) 
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    print("--- Negative Gradient Unlearning Complete ---")
    return model

if __name__ == '__main__':
    # --- Prepare Data ---
    
    # 1. Data for the remaining classes (for fine-tuning)
    retain_train_loader, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE, 
        class_to_forget=CLASS_TO_FORGET
    )

    # 2. Data for the forgotten class (for negative gradient)
    forget_loader = get_forget_loader(CLASS_TO_FORGET, BATCH_SIZE)
    
    # --- Run Unlearning Methods ---

    # Method 1: Fine-tuning
    print("\nExecuting Fine-tuning method...")
    model_ft = load_model(FULL_MODEL_PATH, DEVICE)
    model_ft = unlearn_finetuning(model_ft, DEVICE, retain_train_loader, UNLEARN_EPOCHS, LR)
    torch.save(model_ft.state_dict(), os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ft.pth'))
    print("Fine-tuned unlearned model saved to ./models/model_unlearn_ft.pth")

    # Method 2: Negative Gradient
    print("\nExecuting Negative Gradient method...")
    model_ga = load_model(FULL_MODEL_PATH, DEVICE)
    model_ga = unlearn_negative_gradient(model_ga, DEVICE, forget_loader, UNLEARN_EPOCHS, LR)
    torch.save(model_ga.state_dict(), os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ga.pth'))
    print("Negative Gradient unlearned model saved to ./models/model_unlearn_ga.pth")
    
    print("\n--- All unlearning processes are complete! ---")
