import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
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
    precision = precision_score(y_true, y_pred, average="macro")

    return acc, f1, recall, precision, y_true, y_pred


def print_detailed_metrics(y_true, y_pred, activity_labels):
    """Print detailed evaluation metrics"""
    print("\n" + "=" * 60)
    print("Detailed Performance Analysis")
    print("=" * 60)

    # Classification report
    target_names = list(activity_labels.values())
    report = classification_report(y_true, y_pred,
                                   target_names=target_names, digits=4)
    print(report)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i in range(len(activity_labels)):
        class_mask = np.array(y_true) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(y_pred)[class_mask] == i)
            print(f"{activity_labels[i]:<20}: {class_acc:.4f} ({np.sum(class_mask)} samples)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")

    return cm


def calculate_posture_accuracy(y_true, y_pred, sitting_idx=3, standing_idx=4):
    """Calculate sitting/standing classification accuracy"""
    sitting_mask = np.array(y_true) == sitting_idx
    standing_mask = np.array(y_true) == standing_idx
    posture_mask = sitting_mask | standing_mask

    if np.sum(posture_mask) > 0:
        posture_true = np.array(y_true)[posture_mask]
        posture_pred = np.array(y_pred)[posture_mask]
        posture_acc = accuracy_score(posture_true, posture_pred)

        print(f"\nSitting/Standing Classification Analysis:")
        print(f"  Samples: {np.sum(posture_mask)}")
        print(f"  Accuracy: {posture_acc:.4f}")

        # Detailed analysis
        sitting_true = posture_true == sitting_idx
        standing_true = posture_true == standing_idx

        sitting_pred = posture_pred == sitting_idx
        standing_pred = posture_pred == standing_idx

        sitting_acc = np.mean(sitting_pred[sitting_true])
        standing_acc = np.mean(standing_pred[standing_true])

        print(f"  Sitting accuracy: {sitting_acc:.4f}")
        print(f"  Standing accuracy: {standing_acc:.4f}")

        return posture_acc, sitting_acc, standing_acc

    return None, None, None