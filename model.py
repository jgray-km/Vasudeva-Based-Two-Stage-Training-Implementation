"""
Dual CNN Architecture for Mitigating Simplicity Bias
Based on Vasudeva et al. (2023) - ICLR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight 3-layer CNN for CIFAR-10 subset.
    Designed to extract meaningful representations while being CPU-friendly.
    """
    def __init__(self, num_classes=4, representation_dim=64):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Representation layer (this is Z in the paper)
        self.representation = nn.Linear(64, representation_dim)

        # Classification head
        self.fc = nn.Linear(representation_dim, num_classes)

        self.representation_dim = representation_dim

    def forward(self, x, return_representation=False):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            return_representation: If True, return both logits and representation

        Returns:
            logits: Class predictions [batch_size, num_classes]
            representation (optional): Latent representation [batch_size, representation_dim]
        """
        # Convolutional feature extraction
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = F.relu(self.conv3(x))              # [B, 64, 8, 8]

        # Global average pooling
        x = self.adaptive_pool(x)              # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)              # [B, 64]

        # Get representation (Z)
        z = F.relu(self.representation(x))     # [B, representation_dim]

        # Classification
        logits = self.fc(z)                    # [B, num_classes]

        if return_representation:
            return logits, z
        return logits


class DualModelFramework(nn.Module):
    """
    Dual model framework implementing Vasudeva et al.'s approach.
    Two models are trained jointly with MI regularization to learn complementary features.
    """
    def __init__(self, num_classes=4, representation_dim=64):
        super(DualModelFramework, self).__init__()

        # Two parallel models
        self.model1 = SimpleCNN(num_classes=num_classes, representation_dim=representation_dim)
        self.model2 = SimpleCNN(num_classes=num_classes, representation_dim=representation_dim)

    def forward(self, x, return_representations=False):
        """
        Forward pass through both models.

        Args:
            x: Input tensor
            return_representations: If True, return representations for MI computation

        Returns:
            logits1, logits2: Predictions from both models
            z1, z2 (optional): Representations for MI regularization
        """
        if return_representations:
            logits1, z1 = self.model1(x, return_representation=True)
            logits2, z2 = self.model2(x, return_representation=True)
            return logits1, logits2, z1, z2
        else:
            logits1 = self.model1(x)
            logits2 = self.model2(x)
            return logits1, logits2

    def ensemble_predict(self, x):
        """
        Ensemble prediction by averaging logits from both models.

        Args:
            x: Input tensor

        Returns:
            averaged_logits: Ensemble predictions
        """
        logits1, logits2 = self.forward(x, return_representations=False)
        return (logits1 + logits2) / 2.0


def compute_mi_penalty(z1, z2, method='correlation'):
    """
    Compute mutual information penalty between two representations.

    We use a simplified correlation-based approximation for computational efficiency.
    The goal is to minimize correlation between Z1 and Z2, forcing them to learn
    different features.

    Args:
        z1: Representation from model 1 [batch_size, representation_dim]
        z2: Representation from model 2 [batch_size, representation_dim]
        method: Type of MI approximation ('correlation' or 'cosine')

    Returns:
        mi_penalty: Scalar penalty value (higher = more mutual information)
    """
    if method == 'correlation':
        # Normalize representations
        z1_normalized = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-8)
        z2_normalized = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-8)

        # Compute correlation matrix
        batch_size = z1.size(0)
        correlation = torch.mm(z1_normalized.t(), z2_normalized) / batch_size

        # Penalty is the absolute mean correlation (we want to minimize this)
        mi_penalty = torch.abs(correlation).mean()

    elif method == 'cosine':
        # Cosine similarity between representations
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        # Mean cosine similarity across batch
        cosine_sim = (z1_norm * z2_norm).sum(dim=1).mean()
        mi_penalty = torch.abs(cosine_sim)

    else:
        raise ValueError(f"Unknown MI penalty method: {method}")

    return mi_penalty


if __name__ == "__main__":
    # Test the architecture
    print("Testing Dual Model Framework...")

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)

    # Test single model
    model = SimpleCNN(num_classes=4, representation_dim=64)
    logits = model(x)
    logits_with_rep, z = model(x, return_representation=True)

    print(f"Single model output shape: {logits.shape}")
    print(f"Representation shape: {z.shape}")

    # Test dual framework
    dual_model = DualModelFramework(num_classes=4, representation_dim=64)
    logits1, logits2, z1, z2 = dual_model(x, return_representations=True)

    print(f"\nDual model outputs:")
    print(f"  Model 1 logits: {logits1.shape}")
    print(f"  Model 2 logits: {logits2.shape}")
    print(f"  Model 1 representation: {z1.shape}")
    print(f"  Model 2 representation: {z2.shape}")

    # Test MI penalty
    mi_penalty = compute_mi_penalty(z1, z2)
    print(f"\nMI Penalty (correlation): {mi_penalty.item():.4f}")

    # Test ensemble
    ensemble_logits = dual_model.ensemble_predict(x)
    print(f"Ensemble logits: {ensemble_logits.shape}")

    print("\nArchitecture test passed!")
