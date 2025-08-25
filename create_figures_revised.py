# create_figures_revised.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import các hàm cần thiết từ các file khác
from unlearn import load_model, get_forget_loader

# Suppress common warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
CLASS_TO_FORGET = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = './models'
FIGURE_SAVE_DIR = './figures_revised' # Save to a new directory
DPI = 600

# --- Data from your evaluation run ---
results = {
    "Full Model": {"Retain Acc": 76.07, "Forget Acc": 98.08},
    "Retrain Model": {"Retain Acc": 79.33, "Forget Acc": 0.00},
    "Unlearn (Finetune)": {"Retain Acc": 81.10, "Forget Acc": 72.52},
    "Unlearn (NegGrad)": {"Retain Acc": 52.54, "Forget Acc": 0.04},
}
model_names = list(results.keys())

def plot_annotated_bar_chart():
    """Figure 1 (Revised): Bar chart with ideal reference lines."""
    print("Generating Figure 1: Annotated Bar Chart...")
    
    retain_accs = [res["Retain Acc"] for res in results.values()]
    forget_accs = [res["Forget Acc"] for res in results.values()]
    
    x = np.arange(len(model_names))
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Model Performance Evaluation', fontsize=18, weight='bold')

    # Subplot 1: Retain Accuracy
    colors1 = ['gray', 'green', 'blue', 'orange']
    bars1 = ax1.bar(x, retain_accs, width, color=colors1)
    ax1.set_title('Utility on Retain Classes', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=25, ha="right")
    ax1.bar_label(bars1, fmt='%.2f')
    
    # Add reference line from Retrain Model
    ideal_retain_acc = results['Retrain Model']['Retain Acc']
    ax1.axhline(y=ideal_retain_acc, color='green', linestyle='--', label=f'Ideal (Retrain): {ideal_retain_acc}%')
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Subplot 2: Forget Accuracy
    colors2 = ['red', 'green', 'orange', 'blue']
    bars2 = ax2.bar(x, forget_accs, width, color=colors2)
    ax2.set_title('Performance on Forgotten Class', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=25, ha="right")
    ax2.bar_label(bars2, fmt='%.2f')
    ax2.legend(['Lower is better'])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig1_annotated_bar_chart.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")


def plot_tradeoff_scatter():
    """Figure 2 (New): Trade-off Scatter Plot."""
    print("Generating Figure 2: Trade-off Scatter Plot...")

    forget_accs = [res["Forget Acc"] for res in results.values()]
    retain_accs = [res["Retain Acc"] for res in results.values()]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    palette = sns.color_palette("husl", len(model_names))
    markers = ['o', 's', '^', 'D']

    for i, model in enumerate(model_names):
        ax.scatter(forget_accs[i], retain_accs[i], 
                   label=model, s=200, color=palette[i], 
                   marker=markers[i], alpha=0.8, edgecolors='w')

    # Annotations
    for i, txt in enumerate(model_names):
        ax.annotate(txt, (forget_accs[i] + 1, retain_accs[i] + 0.5), fontsize=10)

    # Ideal region annotation
    ax.text(5, 82, 'Ideal Region', fontsize=14, color='green', ha='left')
    ax.arrow(20, 81.5, -15, 3, head_width=1.5, head_length=2, fc='green', ec='green')

    ax.set_xlabel('Forget Accuracy (%) (Lower is Better →)', fontsize=14)
    ax.set_ylabel('Retain Accuracy (%) (← Higher is Better)', fontsize=14)
    ax.set_title('Unlearning Performance Trade-off', fontsize=16, weight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig2_tradeoff_scatter.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")


def plot_box_plot():
    """Figure 3 (New): Box plot of prediction probabilities on the forgotten class."""
    print("Generating Figure 3: Box Plot...")

    # --- Data Collection ---
    forget_loader = get_forget_loader(class_to_forget=CLASS_TO_FORGET, batch_size=256)
    
    # --- FIXED HERE: Explicitly define the model paths ---
    model_paths = {
        "Full Model": os.path.join(MODEL_SAVE_DIR, 'model_full.pth'),
        "Retrain Model": os.path.join(MODEL_SAVE_DIR, 'model_retrain.pth'),
        "Unlearn (Finetune)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ft.pth'),
        "Unlearn (NegGrad)": os.path.join(MODEL_SAVE_DIR, 'model_unlearn_ga.pth'),
    }

    plot_data = []
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"  - WARNING: Model path not found for {name}. Skipping. Path: {path}")
            continue
            
        print(f"  - Processing {name}...")
        model = load_model(path, DEVICE)
        probs = []
        with torch.no_grad():
            for data, _ in forget_loader:
                data = data.to(DEVICE)
                outputs = model(data)
                softmax_probs = F.softmax(outputs, dim=1)
                probs.extend(softmax_probs[:, CLASS_TO_FORGET].cpu().numpy())
        for p in probs:
            plot_data.append({'Model': name, 'Probability': p})
            
    if not plot_data:
        print("Error: No data was collected for plotting. Please check model paths.")
        return

    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Model', y='Probability', data=df, palette='muted', showfliers=False)
    sns.stripplot(x='Model', y='Probability', data=df, color=".25", size=2, alpha=0.2)

    plt.title("Distribution of Predicted Probability for the Forgotten Class ('cat')", fontsize=16, weight='bold')
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Predicted Probability", fontsize=14)
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURE_SAVE_DIR, 'fig3_box_plot_probability.png')
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    # Set a nice style for the plots
    sns.set_theme(style="whitegrid")

    # Create the directory for figures if it doesn't exist
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    
    # Generate all revised figures
    plot_annotated_bar_chart()
    plot_tradeoff_scatter()
    plot_box_plot()
    
    print("\nAll revised figures have been generated successfully!")
