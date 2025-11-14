"""
Utility functions for visualization and analysis
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history including loss, accuracy, and MI penalty.

    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Model Accuracies
    axes[0, 1].plot(epochs, history['model1_acc'], 'g-', label='Model 1', marker='o')
    axes[0, 1].plot(epochs, history['model2_acc'], 'b-', label='Model 2', marker='s')
    axes[0, 1].plot(epochs, history['ensemble_acc'], 'r-', label='Ensemble', marker='^', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: MI Penalty over time
    axes[1, 0].plot(epochs, history['mi_penalty'], 'purple', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MI Penalty')
    axes[1, 0].set_title('Mutual Information Penalty')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Ensemble vs Individual Models
    axes[1, 1].plot(epochs, history['ensemble_acc'], 'r-', label='Ensemble', linewidth=2)
    axes[1, 1].plot(epochs, history['model1_acc'], 'g--', label='Model 1', alpha=0.6)
    axes[1, 1].plot(epochs, history['model2_acc'], 'b--', label='Model 2', alpha=0.6)
    axes[1, 1].fill_between(epochs, history['model1_acc'], history['ensemble_acc'],
                            alpha=0.2, color='red', label='Ensemble Gain')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Ensemble Improvement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_corruption_results(corruption_results, save_path='corruption_results.png'):
    """
    Plot results on CIFAR-10-C corruptions as a bar chart.

    Args:
        corruption_results: Dictionary mapping corruption -> results
        save_path: Path to save the plot
    """
    corruptions = list(corruption_results.keys())
    model1_accs = [corruption_results[c]['model1_acc'] for c in corruptions]
    model2_accs = [corruption_results[c]['model2_acc'] for c in corruptions]
    ensemble_accs = [corruption_results[c]['ensemble_acc'] for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, model1_accs, width, label='Model 1', color='green', alpha=0.7)
    bars2 = ax.bar(x, model2_accs, width, label='Model 2', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, ensemble_accs, width, label='Ensemble', color='red', alpha=0.7)

    ax.set_xlabel('Corruption Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance on CIFAR-10-C Corruptions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add horizontal line for average
    avg_ensemble = np.mean(ensemble_accs)
    ax.axhline(y=avg_ensemble, color='red', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Avg Ensemble: {avg_ensemble:.1f}%')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Corruption results plot saved to {save_path}")
    plt.close()


def visualize_representations(model, data_loader, device='cpu', num_samples=500,
                               save_path='representations.png'):
    """
    Visualize the learned representations from both models using t-SNE or PCA.

    Args:
        model: Trained DualModelFramework
        data_loader: DataLoader for visualization
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not installed, skipping representation visualization")
        return

    model.eval()

    z1_list = []
    z2_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            _, _, z1, z2 = model(inputs, return_representations=True)

            z1_list.append(z1.cpu().numpy())
            z2_list.append(z2.cpu().numpy())
            labels_list.append(targets.numpy())

            if len(labels_list) * targets.size(0) >= num_samples:
                break

    # Concatenate all batches
    z1_all = np.vstack(z1_list)[:num_samples]
    z2_all = np.vstack(z2_list)[:num_samples]
    labels_all = np.hstack(labels_list)[:num_samples]

    # Apply PCA to reduce to 2D
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)

    z1_2d = pca1.fit_transform(z1_all)
    z2_2d = pca2.fit_transform(z2_all)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Model 1 representations
    scatter1 = axes[0].scatter(z1_2d[:, 0], z1_2d[:, 1], c=labels_all,
                               cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title('Model 1 Representations (PCA)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Class')

    # Model 2 representations
    scatter2 = axes[1].scatter(z2_2d[:, 0], z2_2d[:, 1], c=labels_all,
                               cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('Model 2 Representations (PCA)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Class')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Representation visualization saved to {save_path}")
    plt.close()


def analyze_mi_diversity(model, data_loader, device='cpu', num_batches=10):
    """
    Analyze the diversity between two models by computing MI penalty statistics.

    Args:
        model: Trained DualModelFramework
        data_loader: DataLoader for analysis
        device: Device to run on
        num_batches: Number of batches to analyze

    Returns:
        Dictionary with MI statistics
    """
    from model import compute_mi_penalty

    model.eval()
    mi_penalties = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)
            _, _, z1, z2 = model(inputs, return_representations=True)

            mi_penalty = compute_mi_penalty(z1, z2, method='correlation')
            mi_penalties.append(mi_penalty.item())

    stats = {
        'mean': np.mean(mi_penalties),
        'std': np.std(mi_penalties),
        'min': np.min(mi_penalties),
        'max': np.max(mi_penalties)
    }

    print("\nMI Penalty Statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")

    return stats


def load_checkpoint_and_plot(checkpoint_path='checkpoints/best_model.pth'):
    """
    Load a checkpoint and plot its training history.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history = checkpoint.get('history', None)

    if history is None:
        print("No training history found in checkpoint")
        return

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best accuracy: {checkpoint['accuracy']:.2f}%")

    # Plot training history
    plot_training_history(history, save_path='training_history.png')


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - plot_corruption_results()")
    print("  - visualize_representations()")
    print("  - analyze_mi_diversity()")
    print("  - load_checkpoint_and_plot()")

    # Try to load and plot if checkpoint exists
    if os.path.exists('checkpoints/best_model.pth'):
        print("\nFound checkpoint, generating training history plot...")
        load_checkpoint_and_plot('checkpoints/best_model.pth')
    else:
        print("\nNo checkpoint found. Run train.py first!")
