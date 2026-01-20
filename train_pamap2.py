import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.pamap2_utils.metrics import evaluate_model
from config import config_pamap2


def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, device,
                epochs=config_pamap2.EPOCHS, lr=config_pamap2.LR, model_name="tdn_pamap2"):
    """Train the model"""
    print(f"\nTraining {model_name}...")

    # Create checkpoint directory
    os.makedirs(config_pamap2.SAVE_PATH, exist_ok=True)

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_acc, val_f1, val_recall, _, _ = evaluate_model(model, val_loader, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(config_pamap2.SAVE_PATH, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_path)

        # Track metrics
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        # Print progress
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
              f"Best Val Acc: {best_val_acc:.4f}")

    # Save final model
    model_path = os.path.join(config_pamap2.SAVE_PATH, f"{model_name}_final.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to {config_pamap2.SAVE_PATH}")

    return best_val_acc, train_losses, val_accuracies