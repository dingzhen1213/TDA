import torch
import os
from config import config_pamap2
from utils.pamap2_utils.metrics import evaluate_model


def evaluate_saved_model(model, data_loader, device, model_path):
    """Evaluate a saved model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    acc, f1, recall, y_true, y_pred = evaluate_model(model, data_loader, device)

    print(f"\nEvaluation Results for {os.path.basename(model_path)}:")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")

    return acc, f1, recall, y_true, y_pred


def compare_models(models_dict, data_loader, device, save_dir=config_pamap2.SAVE_PATH):
    """Compare multiple models"""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    results = {}
    for model_name, model in models_dict.items():
        model_path = os.path.join(save_dir, f"{model_name}_best.pth")
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_name}...")
            acc, f1, recall, _, _ = evaluate_saved_model(model, data_loader, device, model_path)
            results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'recall': recall
            }
        else:
            print(f"Model {model_name} not found at {model_path}")

    # Print comparison table
    if results:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Recall':<10}")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} {metrics['recall']:<10.4f}")

    return results