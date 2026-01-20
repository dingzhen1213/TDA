import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from config import config_WISDM
from utils.WISDM_utils.data_loader import load_wisdm_dataset
from models.WISDM_models.tdn import  TDN_WISDM
from train_WISDM import train_model
from evaluate_WISDM import evaluate_model_with_analysis


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=config_WISDM.BATCH_SIZE):
    """Create train, validation, and test data loaders"""
    # Split training data into train and validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=config_WISDM.VAL_SIZE,
        random_state=config_WISDM.SEED,
        stratify=y_train
    )

    def to_tensor_dataset(X, y):
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T)
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    # Create datasets
    train_dataset = to_tensor_dataset(X_tr, y_tr)
    val_dataset = to_tensor_dataset(X_val, y_val)
    test_dataset = to_tensor_dataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data loaders created:")
    print(f"  Training: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="TDN for WISDM")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--window_size", type=int, default=config_WISDM.WINDOW_SIZE,
                        help="Window size")
    parser.add_argument("--batch_size", type=int, default=config_WISDM.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config_WISDM.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config_WISDM.LR,
                        help="Learning rate")
    parser.add_argument("--split_by_user", action="store_true",
                        help="Split data by user")
    parser.add_argument("--model_path", type=str,
                        default="save/WISDM/checkpoints/tdn_wisdm_best.pth",
                        help="Path to model checkpoint")

    args = parser.parse_args()

    print("=" * 60)
    print("TDN-WISDM: Temporal Decomposition Network")
    print("=" * 60)

    # Load data
    print("\nLoading WISDM dataset...")
    try:
        X_train, y_train, X_test, y_test, label_encoder = load_wisdm_dataset(
            data_path=config_WISDM.WISDM_DATA_PATH,
            split_by_user=args.split_by_user,
            window_size=args.window_size,
            overlap=config_WISDM.OVERLAP
        )
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure the WISDM dataset file is in the correct location.")
        print(f"Expected file: {config_WISDM.WISDM_DATA_PATH}")
        print("You can download it from: https://www.cis.fordham.edu/wisdm/dataset.php")
        return

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Window size: {X_train.shape[1]}")
    print(f"  Number of channels: {X_train.shape[2]}")
    print(f"  Activity classes: {list(label_encoder.classes_)}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size
    )

    # Initialize model
    device = config_WISDM.DEVICE
    input_channels = X_train.shape[2]
    num_classes = len(label_encoder.classes_)

    model = TDN_WISDM(
        window_size=args.window_size,
        input_channels=input_channels,
        num_classes=num_classes,
        embed_dim=config_WISDM.EMBED_DIM,
        feature_dim=config_WISDM.FEATURE_DIM,
        use_multi_scale=config_WISDM.USE_MULTI_SCALE
    ).to(device)

    print(f"\nModel Configuration:")
    print(f"  Device: {device}")
    print(f"  Input channels: {input_channels}")
    print(f"  Number of classes: {num_classes}")

    # Training
    if args.train:
        print(f"\n{'=' * 60}")
        print("Training Phase")
        print(f"{'=' * 60}")

        best_acc, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            model_name="tdn_wisdm"
        )

    # Evaluation
    if args.eval:
        print(f"\n{'=' * 60}")
        print("Evaluation Phase")
        print(f"{'=' * 60}")

        results = evaluate_model_with_analysis(
            model=model,
            test_loader=test_loader,
            device=device,
            label_encoder=label_encoder,
            model_path=args.model_path
        )

        # Print final summary
        print(f"\n{'=' * 60}")
        print("FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Test Accuracy:      {results['accuracy']:.4f}")
        print(f"Test F1 Score:      {results['f1_score']:.4f}")
        print(f"Test Recall:        {results['recall']:.4f}")
        print(f"Test Precision:     {results['precision']:.4f}")

    if not args.train and not args.eval:
        print("No action specified. Use --train to train or --eval to evaluate.")


if __name__ == "__main__":
    main()