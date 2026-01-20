import argparse
from config import config_UCI
from utils.UCIHAR_utils.dataloader import load_uci_raw_inertial_data, preprocess_raw_data, create_data_loaders
from models.UCIHAR_models.tdn import TDN_UCI
from train_UCI import train_model
from evaluate_UCI import evaluate_model_with_analysis


def main():
    parser = argparse.ArgumentParser(description="TDN for UCI-HAR")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--window_size", type=int, default=config_UCI.WINDOW_SIZE,
                        help="Window size")
    parser.add_argument("--batch_size", type=int, default=config_UCI.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config_UCI.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config_UCI.LR,
                        help="Learning rate")
    parser.add_argument("--model_path", type=str,
                        default="save/UCIHAR/checkpoints/tdn_uci_best.pth",
                        help="Path to model checkpoint")

    args = parser.parse_args()

    print("=" * 60)
    print("TDN-UCI-HAR: Temporal Decomposition Network")
    print("=" * 60)

    # Load data
    print("\nLoading UCI-HAR dataset...")
    try:
        X_train, X_test, y_train, y_test = load_uci_raw_inertial_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure the UCI HAR Dataset is in the correct location.")
        print(f"Expected path: {config_UCI.UCI_DATA_PATH}")
        print(
            "You can download it from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones")
        return

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Window size: {X_train.shape[1]}")
    print(f"  Number of channels: {X_train.shape[2]}")
    print(f"  Activity labels: {list(config_UCI.ACTIVITY_LABELS.values())}")

    # Preprocess data
    X_train, X_test = preprocess_raw_data(X_train, X_test)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size
    )

    # Initialize model
    device = config_UCI.DEVICE
    input_channels = X_train.shape[2]
    num_classes = len(config_UCI.ACTIVITY_LABELS)

    model = TDN_UCI(
        window_size=args.window_size,
        input_channels=input_channels,
        num_classes=num_classes,
        embed_dim=config_UCI.EMBED_DIM,
        feature_dim=config_UCI.FEATURE_DIM,
        use_multi_scale=config_UCI.USE_MULTI_SCALE
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
            model_name="tdn_uci"
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
            model_path=args.model_path
        )

        # Print final summary
        print(f"\n{'=' * 60}")
        print("FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"Test Accuracy:      {results['accuracy']:.4f}")
        print(f"Test F1 Score:      {results['f1_score']:.4f}")
        if results['posture_accuracy'] is not None:
            print(f"Sitting/Standing Acc: {results['posture_accuracy']:.4f}")

    if not args.train and not args.eval:
        print("No action specified. Use --train to train or --eval to evaluate.")


if __name__ == "__main__":
    main()