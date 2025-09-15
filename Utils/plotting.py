from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import argparse
import numpy as np
import os
import random

def plot_metrics(train_metrics, train_losses, val_metrics, val_losses):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(train_metrics, label='Train')
    ax[0].plot(val_metrics, label='Validation')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(train_losses, label='Train')
    ax[1].plot(val_losses, label='Validation')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

def plot_csv_alpha_mean_loyalty(file:str) -> Figure:
    df = pd.read_csv(file)
    fig, ax = plt.subplots()
    for name, group in df.groupby("Type"):
        ax.plot(group["Alpha"], group["Mean Loyalty"], label=name)
    ax.set_title("Mean Loyalty by Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Loyalty")
    ax.legend()
    return fig

def plot_csv_complexity_mean_loyalty(file:str) -> Figure:
    df = pd.read_csv(file)
    representation_type = ["o", "x", '+', "|", "s"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, group) in enumerate(df.groupby("Type")):
        scatter = ax.scatter(group["Complexity"], group["Mean Loyalty"], 
                            label=name, c=group['Alpha'], cmap='viridis', 
                            marker=representation_type[i])
    
    ax.set_title(f"Mean Loyalty")
    ax.set_xlabel("Complexity\n(Num Segments)")
    ax.set_ylabel("Mean Loyalty")
    
    min_complexity = df["Complexity"].min()
    max_complexity = df["Complexity"].max()
    num_ticks = 6
    complexity_ticks = np.linspace(min_complexity, max_complexity, num_ticks)
    
    labels = []
    for comp in complexity_ticks:
        closest_comp = df["Complexity"].iloc[(df["Complexity"] - comp).abs().argsort()[:1]].values[0]
        segments = sorted(df[df["Complexity"] == closest_comp]["Num Segments"].unique())
        segments_str = ", ".join(map(str, segments))
        labels.append(f"{comp:.1f}\n({segments_str})")
    
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(labels)
    
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha/Epsilon')
    plt.tight_layout()
    
    return fig

def plot_csv_complexity_kappa_loyalty(file:str, points:dict={}) -> Figure:
    assert isinstance(points, dict), "Points should be a dictionary"
    assert isinstance(file, str), "File should be a string"
    assert os.path.exists(file), f"File {file} does not exist"
    assert file.endswith(".csv"), "File should be a CSV file"
    
    df = pd.read_csv(file)
    representation_type = ["o", "x", '+', "|", "s"]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prefer Percentage Agreement as Loyalty; fallback to Mean Loyalty or Kappa Loyalty
    loyalty_col = None
    if "Percentage Agreement" in df.columns:
        loyalty_col = "Percentage Agreement"
    elif "Mean Loyalty" in df.columns:
        loyalty_col = "Mean Loyalty"
    elif "Kappa Loyalty" in df.columns:
        loyalty_col = "Kappa Loyalty"
    else:
        raise ValueError("CSV missing loyalty column (Percentage Agreement/Mean Loyalty/Kappa Loyalty)")

    for i, (name, group) in enumerate(df.groupby("Type")):
        y_vals = group[loyalty_col]
        # If loyalty is not a percentage, scale to percentage for display
        if loyalty_col != "Percentage Agreement":
            try:
                y_vals = y_vals.astype(float) * 100.0
            except Exception:
                y_vals = y_vals
        scatter = ax.scatter(group["Complexity"], y_vals,
                             label=name, c=group['Alpha'], cmap='viridis',
                             marker=representation_type[i])
        
        if points != {}:
            point_x = float(points[name][0])
            point_y = float(points[name][1])
            ax.scatter(point_x, point_y, color='red', marker=representation_type[i])
            #ax.axhline(y=point_y, color='red', linestyle='--', alpha=0.2)
            #ax.axvline(x=point_x, color='red', linestyle='--', alpha=0.2)
    
    ax.set_title(f"Loyalty")
    ax.set_xlabel("Complexity\n(Abs. Num. Segments)")
    ax.set_ylabel("Loyalty (%)")
    
    min_complexity = df["Complexity"].min()
    max_complexity = df["Complexity"].max()
    num_ticks = 6
    complexity_ticks = np.linspace(min_complexity, max_complexity, num_ticks)
    
    x_labels = []
    for comp in complexity_ticks:
        closest_comp = df["Complexity"].iloc[(df["Complexity"] - comp).abs().argsort()[:1]].values[0]
        segments = sorted(df[df["Complexity"] == closest_comp]["Num Segments"].unique())
        segments_str = ", ".join(map(str, segments))
        x_labels.append(f"{comp:.1f}\n({segments_str})")

    # Configure Y ticks in percentage space
    try:
        if loyalty_col == "Percentage Agreement":
            y_min, y_max = df[loyalty_col].min(), df[loyalty_col].max()
        else:
            # Scale to percentage for tick calculation
            y_min, y_max = df[loyalty_col].astype(float).min() * 100.0, df[loyalty_col].astype(float).max() * 100.0
        loyalty_ticks = np.linspace(y_min, y_max, num_ticks)
        ax.set_yticks(loyalty_ticks)
        ax.set_yticklabels([f"{t:.0f}%" for t in loyalty_ticks])
    except Exception:
        pass
    
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(x_labels)

    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Alpha/Epsilon')
    plt.tight_layout()
    
    return fig


def plot_prototipes(alpha: np.float64, pred_class: np.int8, X: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X)
    ax.set_title(f"Prototypes for Alpha: {alpha}, Predicted Class: {pred_class}")
    return fig

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, help="File to plot")
    args = argparser.parse_args()
    file = args.file
    assert os.path.exists(file), f"File {file} does not exist"

    if file.endswith(".csv"):
        fig1 = plot_csv_complexity_mean_loyalty(file)
        plt.show()
        fig2 = plot_csv_complexity_kappa_loyalty(file)
        plt.show()
    elif file.endswith(".npy"):
        data = np.load(file)
        rand_num = random.randint(0, len(data) - 1)
        pred_classes = [data[i][1] for i in range(len(data))]
        unique_classes = np.unique(pred_classes)
        print(f"Unique classes: {unique_classes}")
        for i in range(10):
            rand_num = random.randint(1,99)
            alpha = data[rand_num][0]; pred_class = data[rand_num][1]; X = data[rand_num][-1]
            fig = plot_prototipes(alpha, pred_class, X)
            plt.show()
    else:
        raise ValueError(f"File format not supported")
