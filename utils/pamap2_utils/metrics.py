import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, data_loader, device):
    """Evaluate model on given data loader"""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions.tolist())
            y_true.extend(y_batch.numpy().tolist())

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    return acc, f1, recall, y_true, y_pred


def print_detailed_metrics(y_true, y_pred, label_encoder):
    """Print detailed evaluation metrics"""
    print("\n" + "=" * 60)
    print("Detailed Performance Analysis")
    print("=" * 60)

    # Classification report
    class_names = [f'Class {i}' for i in label_encoder.classes_]
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4)
    print(report)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i in range(len(label_encoder.classes_)):
        class_mask = np.array(y_true) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(y_pred)[class_mask] == i)
            print(f"Class {label_encoder.classes_[i]} (ID {i}): {class_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")

    return cm