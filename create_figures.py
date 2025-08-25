# create_figures.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from unlearn import load_model, get_forget_loader

# Suppress the specific FutureWarning from torch.load
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# --- Configuration ---
CLASS_TO_FORGET = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = './models'
FIGURE_SAVE_DIR = './figures'
DPI = 600

# --- Data from your evaluation run ---
# I'm hardcoding the results here for simplicity.
# You can also make the script re-calculate them if you prefer.
results = {
    "Full Model": {"Retain Acc": 76.07, "Forget Acc": 98.08},
    "Retrain Model": {"Retain Acc": 79.33, "Forget Acc": 0.00},
    "Unlearn (Finetune)": {"Retain Acc": 81.10, "Forget Acc": 72.52},
    "Unlearn (NegGrad)": {"Retain Acc": 52.54, "Forget Acc": 0.04},
}
model_names = list(results.keys())
retain_accs = [res["Retain Acc"] for res in results.values()]
forget_accs = [res["Forget Acc"] for res in results.values()]


def plot_bar_chart():
    """Figure 1: Grouped Bar Chart for direct comparison."""
    print("Generating Figure 1: Grouped Bar Chart...")
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, retain_accs, width, label='Retain Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, forget_accs, width, label='Forget Accuracy', color='salmon')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Comparison of Unlearning Methods', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    ax.set_ylim(0, 110) # Set Y-axis limit to give some space
    fig.tight_layout()
    
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig1_bar_chart_comparison.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")


def plot_radar_chart():
    """Figure 2: Radar Chart for performance profiles."""
    print("Generating Figure 2: Radar Chart...")
    
    labels = ['Retain Accuracy', 'Forget Efficacy']
    num_vars = len(labels)
    
    # Calculate angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Forget Efficacy is 100 - Forget Acc
    data = []
    for model in model_names:
        retain_val = results[model]['Retain Acc']
        forget_efficacy_val = 100 - results[model]['Forget Acc']
        values = [retain_val, forget_efficacy_val]
        values += values[:1] # Complete the loop
        data.append(values)
        
    for i, model in enumerate(model_names):
        ax.plot(angles, data[i], label=model, linewidth=2)
        ax.fill(angles, data[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_rlabel_position(180 / num_vars)
    ax.set_ylim(0, 100)

    plt.title('Performance Profile of Models', size=16, color='black', y=1.1, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig2_radar_chart_profile.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")

def get_forget_class_probabilities(model, data_loader, device):
    """Helper function to run inference and get probabilities for the forget class."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            # Get probabilities for the forget class (index 3)
            probabilities.extend(probs[:, CLASS_TO_FORGET].cpu().numpy())
    return probabilities

def plot_violin_plot():
    """Figure 3: Violin plot of prediction probabilities on the forgotten class."""
    print("Generating Figure 3: Violin Plot (this may take a moment)...")

    # --- Data Collection ---
    forget_loader = get_forget_loader(class_to_forget=CLASS_TO_FORGET, batch_size=256)
    
    model_paths = {
        "Full Model": os.path.join(MODEL_SAVE_DIR, 'model_full.pth'),
        "Retrain Model": os.path.join(MODEL_SAVE_DIR, 'model_retrain.pth'),
        "Unlearn (Finetune)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ft.pth'),
        "Unlearn (NegGrad)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ga.pth'),
    }
    
    plot_data = []
    for name, path in model_paths.items():
        print(f"  - Processing {name}...")
        model = load_model(path, DEVICE)
        probs = get_forget_class_probabilities(model, forget_loader, DEVICE)
        for p in probs:
            plot_data.append({'Model': name, 'Probability': p})
            
    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Model', y='Probability', data=df, inner='quartile', palette='muted')
    
    plt.title("Distribution of Predicted Probability for the Forgotten Class ('cat')", fontsize=16, weight='bold')
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Predicted Probability", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig3_violin_plot_probability.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    # Set a nice style for the plots
    sns.set_theme(style="whitegrid")

    # Create the directory for figures if it doesn't exist
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    
    # Generate all figures
    plot_bar_chart()
    plot_radar_chart()
    plot_violin_plot()
    
    print("\nAll figures have been generated successfully!")
