import argparse
from config import config_pamap2
from utils.pamap2_utils.data_loader import load_pamap2_dataset, create_dataloaders, set_seed
from models.pamap2_models.tdn import TDN_PAMAP2
from train_pamap2 import train_model
from evaluate_pamap2 import evaluate_saved_model


def main():
    parser = argparse.ArgumentParser(description="TDN for PAMAP2 HAR")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--window_size", type=int, default=config_pamap2.WINDOW_SIZE,
                        help="Window size for sliding windows")
    parser.add_argument("--batch_size", type=int, default=config_pamap2.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=config_pamap2.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config_pamap2.LR,
                        help="Learning rate")
    parser.add_argument("--use_18ch", action="store_true",
                        help="Use 18-channel config_pamap2uration")
    parser.add_argument("--split_by_user", action="store_true",
                        help="Split data by user")
    parser.add_argument("--model_path", type=str,
                        default="save/pamap2/checkpoints/tdn_pamap2_best.pth",
                        help="Path to model checkpoint")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(config_pamap2.SEED)

    # Load data
    print("=" * 60)
    print("TDN-PAMAP2: Temporal Decomposition Network for HAR")
    print("=" * 60)

    data = load_pamap2_dataset(
        data_path=config_pamap2.DATA_PATH,
        window_size=args.window_size,
        overlap=config_pamap2.OVERLAP,
        use_18_channels=args.use_18ch or config_pamap2.USE_18_CHANNELS,
        split_by_user=args.split_by_user or config_pamap2.SPLIT_BY_USER,
        test_size=config_pamap2.TEST_SIZE,
        random_state=config_pamap2.RANDOM_STATE
    )

    X_train, y_train, X_test, y_test, label_encoder, _, _ = data

    print(f"\nDataset Statistics:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Window size: {X_train.shape[1]}")
    print(f"Number of channels: {X_train.shape[2]}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=args.batch_size
    )

    # Initialize model
    device = config_pamap2.DEVICE
    input_channels = X_train.shape[2]
    num_classes = len(label_encoder.classes_)

    model = TDN_PAMAP2(
        window_size=args.window_size,
        input_channels=input_channels,
        num_classes=num_classes,
        embed_dim=config_pamap2.EMBED_DIM,
        feature_dim=config_pamap2.FEATURE_DIM,
        use_multi_scale=config_pamap2.USE_MULTI_SCALE
    ).to(device)

    print(f"\nModel initialized on {device}")
    print(f"Input channels: {input_channels}")
    print(f"Number of classes: {num_classes}")

    # Training
    if args.train:
        best_acc, train_losses, val_accuracies = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            model_name="tdn_pamap2"
        )

    # Evaluation
    if args.eval:
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)

        acc, f1, recall, y_true, y_pred = evaluate_saved_model(
            model=model,
            data_loader=test_loader,
            device=device,
            model_path=args.model_path
        )
        # Save predictions
        import pandas as pd
        predictions_df = pd.DataFrame({
            'true': y_true,
            'predicted': y_pred,
            'true_label': label_encoder.inverse_transform(y_true),
            'predicted_label': label_encoder.inverse_transform(y_pred)
        })
        predictions_df.to_csv("predictions.csv", index=False)
        print("\nPredictions saved to predictions.csv")

    if not args.train and not args.eval:
        print("No action specified. Use --train to train or --eval to evaluate.")


if __name__ == "__main__":
    main()