import torch
import os
import config
from utils.UCIHAR_utils.metrics import evaluate_model, print_detailed_metrics, calculate_posture_accuracy


def evaluate_saved_model(model, data_loader, device, model_path):
    """Evaluate a saved model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    acc, f1, recall, precision, y_true, y_pred = evaluate_model(model, data_loader, device)

    print(f"\nEvaluation Results for {os.path.basename(model_path)}:")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    return acc, f1, recall, precision, y_true, y_pred


def evaluate_model_with_analysis(model, test_loader, device, model_path="save/UCIHAR/checkpoints/tdn_uci_best.pth"):
    """Evaluate model and provide detailed analysis"""
    print("\n" + "=" * 60)
    print("Model Evaluation and Analysis")
    print("=" * 60)

    # Evaluate model
    acc, f1, recall, precision, y_true, y_pred = evaluate_saved_model(
        model, test_loader, device, model_path
    )

    # Print detailed metrics
    cm = print_detailed_metrics(y_true, y_pred, config.ACTIVITY_LABELS)

    # Calculate sitting/standing accuracy
    posture_acc, sitting_acc, standing_acc = calculate_posture_accuracy(y_true, y_pred)

    # Print confusion matrix values
    print(f"\nConfusion Matrix Values:")
    for i in range(len(config.ACTIVITY_LABELS)):
        row_values = " ".join(f"{val:4d}" for val in cm[i])
        print(f"  {row_values}")

    return {
        'accuracy': acc,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'posture_accuracy': posture_acc,
        'sitting_accuracy': sitting_acc,
        'standing_accuracy': standing_acc,
        'confusion_matrix': cm
    }