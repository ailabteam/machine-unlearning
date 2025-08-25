# evaluate.py
import torch
import torch.nn as nn
import torchvision.models as models
from prettytable import PrettyTable
import os

from utils import get_cifar10_loaders
from unlearn import get_forget_loader, load_model

# --- Configuration ---
BATCH_SIZE = 256
CLASS_TO_FORGET = 3 # Class 'cat'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = './models'

# --- Evaluation Function ---
def evaluate_model(model, device, test_loader, description=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"{description:<25} | Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return accuracy

if __name__ == '__main__':
    print("--- Starting Evaluation ---")
    
    # --- Prepare Data Loaders ---
    # 1. Loader for the 9 Retain classes
    _, retain_test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE, 
        class_to_forget=CLASS_TO_FORGET
    )
    
    # 2. Loader for the Forget class only
    forget_test_loader = get_forget_loader(
        class_to_forget=CLASS_TO_FORGET,
        batch_size=BATCH_SIZE
    )
    
    # 3. Loader for all 10 classes (for context)
    _, full_test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    # --- Load Models ---
    model_paths = {
        "Full Model": os.path.join(MODEL_SAVE_DIR, 'model_full.pth'),
        "Retrain Model": os.path.join(MODEL_SAVE_DIR, 'model_retrain.pth'),
        "Unlearn (Finetune)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ft.pth'),
        "Unlearn (NegGrad)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ga.pth'),
    }

    results = {}

    for name, path in model_paths.items():
        print(f"\n--- Evaluating: {name} ---")
        if not os.path.exists(path):
            print(f"Warning: Model path not found for {name}. Skipping.")
            continue
            
        model = load_model(path, DEVICE)
        
        # Metric 1: Accuracy on Retain Set (9 classes)
        retain_acc = evaluate_model(model, DEVICE, retain_test_loader, "Accuracy on Retain Set")
        
        # Metric 2: Accuracy on Forget Set (1 class)
        forget_acc = evaluate_model(model, DEVICE, forget_test_loader, "Accuracy on Forget Set")
        
        results[name] = {
            "Retain Acc (%)": f"{retain_acc:.2f}",
            "Forget Acc (%)": f"{forget_acc:.2f}"
        }

    # --- Print Summary Table ---
    print("\n\n--- Evaluation Summary ---")
    table = PrettyTable()
    table.field_names = ["Model", "Retain Acc (%)", "Forget Acc (%)"]
    
    for name, res in results.items():
        table.add_row([name, res["Retain Acc (%)"], res["Forget Acc (%)"]])
        
    print(table)
    
    print("\n--- Analysis ---")
    print("1. Retain Acc: How well the model performs on the tasks it's supposed to. Higher is better. 'Retrain Model' is the gold standard.")
    print("2. Forget Acc: How well the model remembers the forgotten class. Lower is better. Ideally close to random guess (10% for 10 classes, but can be lower).")
