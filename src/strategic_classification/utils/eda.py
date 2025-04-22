import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_burden_vs_accuracy(path, model_name, baseline_accuracy=None, figsize=(8, 6)):
    """
    Plots the tradeoff between burden and accuracy for different lambda values.
    
    Args:
        path (str): Root directory containing subfolders with val_burdens.csv and val_errors.csv.
        baseline_accuracy (float, optional): Reference accuracy to plot as horizontal line.
        figsize (tuple): Size of the figure.

    Returns:
        None
    """

    results = []
    accuracies = []
    lambdas = []

    for root, d_names,f_names in os.walk(path):
        if f"val_{model_name}s.csv" in f_names:
            lamb = root.split("/")[-1].split("_")[-1]
            lambdas.append(lamb)
            val_burdens = pd.read_csv(root + f'/val_{model_name}s.csv')
            val = val_burdens.values[-1][1]
            results.append(val)
            val_errors = pd.read_csv(root + '/val_errors.csv')
            acc = 1 - val_errors.values[-1][1]
            accuracies.append(acc)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(results, accuracies, color='blue')
    ax.set_title(f'{model_name} vs Accuracy Tradeoff')
    ax.set_xlabel(model_name)
    ax.set_ylabel('Accuracy')
    plt.title(f'{model_name} VS accuracy tradeoff')
    plt.xlabel(model_name.title())
    plt.ylabel('Accuracy')
    ax.axhline(y=baseline_accuracy, color='red', linestyle='--', label='Baseline Accuracy')
    ax.legend()

    red_patch = mpatches.Patch(color='r', label='Non strategic data & non strategic model')
    blue_patch = mpatches.Patch(color='b', label='Strategic data & strategic model')
    plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_burden_vs_accuracy("./models", model_name="burden", baseline_accuracy=0.71)
