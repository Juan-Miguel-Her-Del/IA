"""Metricas y graficas; no selecciona hiperparametros con el conjunto test."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from utils.config import CLASSES


def scores(truth, predictions):
    return {"accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": float(f1_score(truth, predictions, labels=list(range(6)), average="macro", zero_division=0))}


def final_metrics(truth, predictions):
    return {**scores(truth, predictions),
            "per_class": classification_report(truth, predictions, labels=list(range(6)),
                                               target_names=list(CLASSES), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(6))).tolist(),
            "class_order": list(CLASSES)}


def plot_history(history, path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"), ("Loss", "Accuracy", "Macro F1")):
        for split, label in (("train", "Entrenamiento"), ("val", "Validacion")):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=label)
        axis.set(xlabel="Epoca", title=title)
        if metric != "loss":
            axis.set_ylim(0, 1)
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(Path(path), dpi=140)
    plt.close(figure)


def plot_confusion(matrix, path, class_names=None, title="Matriz de confusion - prueba"):
    class_names = class_names or CLASSES
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(6), yticks=range(6), xticklabels=class_names, yticklabels=class_names,
             xlabel="Prediccion", ylabel="Categoria real", title=title)
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(6):
        for column in range(6):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
