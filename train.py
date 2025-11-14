"""
Training script for dual model framework with MI regularization
Based on Vasudeva et al. (2023) - Mitigating Simplicity Bias
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm

from model import DualModelFramework, compute_mi_penalty
from dataset import CIFAR10Subset, get_cifar10_class_names


class DualModelTrainer:
    """
    Trainer for the dual model framework with MI regularization.
    """
    def __init__(self, num_classes=4, representation_dim=64, lambda_mi=1.0,
                 learning_rate=1e-3, device='cpu'):
        """
        Args:
            num_classes: Number of output classes
            representation_dim: Dimension of latent representation
            lambda_mi: Weight for MI regularization penalty
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.device = device
        self.lambda_mi = lambda_mi
        self.num_classes = num_classes

        # Initialize dual model
        self.model = DualModelFramework(
            num_classes=num_classes,
            representation_dim=representation_dim
        ).to(device)

        # Optimizer for both models
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'mi_penalty': [],
            'model1_acc': [],
            'model2_acc': [],
            'ensemble_acc': []
        }

        print(f"Initialized DualModelTrainer:")
        print(f"  Device: {device}")
        print(f"  Lambda MI: {lambda_mi}")
        print(f"  Learning rate: {learning_rate}")

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            avg_loss, avg_acc, avg_mi_penalty
        """
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_mi_penalty = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass through both models
            logits1, logits2, z1, z2 = self.model(inputs, return_representations=True)

            # Task losses (classification)
            loss1 = self.criterion(logits1, targets)
            loss2 = self.criterion(logits2, targets)
            task_loss = loss1 + loss2

            # MI regularization penalty
            mi_penalty = compute_mi_penalty(z1, z2, method='correlation')

            # Total loss
            total_loss_batch = task_loss + self.lambda_mi * mi_penalty

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            # Ensemble accuracy
            ensemble_logits = (logits1 + logits2) / 2.0
            _, predicted = ensemble_logits.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

            total_loss += total_loss_batch.item()
            total_mi_penalty += mi_penalty.item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = 100.0 * total_correct / total_samples
        avg_mi_penalty = total_mi_penalty / len(train_loader)

        return avg_loss, avg_acc, avg_mi_penalty

    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        Evaluate on test/validation data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with metrics for model1, model2, and ensemble
        """
        self.model.eval()

        total_samples = 0
        correct_model1 = 0
        correct_model2 = 0
        correct_ensemble = 0
        total_loss = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            logits1, logits2 = self.model(inputs, return_representations=False)

            # Ensemble prediction
            ensemble_logits = (logits1 + logits2) / 2.0

            # Loss (using ensemble)
            loss = self.criterion(ensemble_logits, targets)
            total_loss += loss.item()

            # Accuracy for each model
            _, pred1 = logits1.max(1)
            _, pred2 = logits2.max(1)
            _, pred_ensemble = ensemble_logits.max(1)

            correct_model1 += pred1.eq(targets).sum().item()
            correct_model2 += pred2.eq(targets).sum().item()
            correct_ensemble += pred_ensemble.eq(targets).sum().item()

            total_samples += targets.size(0)

        results = {
            'loss': total_loss / len(test_loader),
            'model1_acc': 100.0 * correct_model1 / total_samples,
            'model2_acc': 100.0 * correct_model2 / total_samples,
            'ensemble_acc': 100.0 * correct_ensemble / total_samples
        }

        return results

    def train(self, train_loader, test_loader, num_epochs=10, save_dir='checkpoints'):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print("="*70)

        os.makedirs(save_dir, exist_ok=True)
        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_acc, mi_penalty = self.train_epoch(train_loader)

            # Evaluate
            val_results = self.evaluate(test_loader)

            epoch_time = time.time() - start_time

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_acc'].append(val_results['ensemble_acc'])
            self.history['mi_penalty'].append(mi_penalty)
            self.history['model1_acc'].append(val_results['model1_acc'])
            self.history['model2_acc'].append(val_results['model2_acc'])
            self.history['ensemble_acc'].append(val_results['ensemble_acc'])

            # Print progress
            print(f"Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, MI Penalty={mi_penalty:.4f}")
            print(f"  Val:   Loss={val_results['loss']:.4f}")
            print(f"         Model1={val_results['model1_acc']:.2f}%, "
                  f"Model2={val_results['model2_acc']:.2f}%, "
                  f"Ensemble={val_results['ensemble_acc']:.2f}%")
            print("-"*70)

            # Save best model
            if val_results['ensemble_acc'] > best_acc:
                best_acc = val_results['ensemble_acc']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, best_acc
                )
                print(f"  ✓ Saved best model (acc={best_acc:.2f}%)")

        print("="*70)
        print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")

        # Save final model
        self.save_checkpoint(
            os.path.join(save_dir, 'final_model.pth'),
            num_epochs, val_results['ensemble_acc']
        )

    def save_checkpoint(self, path, epoch, acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': acc,
            'lambda_mi': self.lambda_mi,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"(acc={checkpoint['accuracy']:.2f}%)")
        return checkpoint


def main():
    """Main training function."""
    # Configuration
    SELECTED_CLASSES = [0, 1, 2, 3]  # airplane, automobile, bird, cat, dog
    NUM_CLASSES = len(SELECTED_CLASSES)
    REPRESENTATION_DIM = 64
    LAMBDA_MI = 0.5  # Weight for MI regularization
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    DEVICE = 'cpu'

    # Print configuration
    class_names = get_cifar10_class_names()
    selected_names = [class_names[i] for i in SELECTED_CLASSES]

    print("="*70)
    print("Dual Model Training - Mitigating Simplicity Bias")
    print("="*70)
    print(f"Classes: {selected_names}")
    print(f"Lambda MI: {LAMBDA_MI}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*70)

    # Load dataset
    print("\nLoading CIFAR-10 subset...")
    cifar_subset = CIFAR10Subset(selected_classes=SELECTED_CLASSES, download=True)
    train_loader = cifar_subset.get_train_loader(batch_size=BATCH_SIZE)
    test_loader = cifar_subset.get_test_loader(batch_size=BATCH_SIZE)

    # Initialize trainer
    trainer = DualModelTrainer(
        num_classes=NUM_CLASSES,
        representation_dim=REPRESENTATION_DIM,
        lambda_mi=LAMBDA_MI,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        save_dir='checkpoints'
    )

    print("\nTraining script completed successfully!")


if __name__ == "__main__":
    main()
