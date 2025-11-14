"""
Evaluation script for testing on CIFAR-10-C corruptions
Demonstrates improved OOD generalization with dual model framework
"""

import torch
import numpy as np
import os
from tabulate import tabulate

from model import DualModelFramework
from dataset import CIFAR10CCorrupted, CIFAR10Subset, get_cifar10_class_names


class ModelEvaluator:
    """
    Evaluate trained models on CIFAR-10-C corruptions.
    """
    def __init__(self, checkpoint_path, selected_classes=None, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            selected_classes: Classes used during training (auto-detected if None)
            device: Device to run evaluation on
        """
        self.device = device

        # Load checkpoint first to detect number of classes
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Auto-detect number of classes from model weights
        if selected_classes is None:
            # Get num_classes from the final layer shape
            num_classes = checkpoint['model_state_dict']['model1.fc.bias'].shape[0]
            print(f"  Auto-detected {num_classes} classes from checkpoint")

            # Use first N classes as default
            selected_classes = list(range(num_classes))

        self.selected_classes = selected_classes
        self.num_classes = len(selected_classes)

        # Initialize model with correct size
        self.model = DualModelFramework(
            num_classes=self.num_classes,
            representation_dim=64
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"  Classes: {self.num_classes}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Training accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")

    @torch.no_grad()
    def evaluate_loader(self, data_loader):
        """
        Evaluate on a single data loader.

        Args:
            data_loader: DataLoader to evaluate

        Returns:
            Dictionary with accuracy metrics
        """
        if data_loader is None:
            return None

        total_samples = 0
        correct_model1 = 0
        correct_model2 = 0
        correct_ensemble = 0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            logits1, logits2 = self.model(inputs, return_representations=False)

            # Ensemble prediction
            ensemble_logits = (logits1 + logits2) / 2.0

            # Predictions
            _, pred1 = logits1.max(1)
            _, pred2 = logits2.max(1)
            _, pred_ensemble = ensemble_logits.max(1)

            correct_model1 += pred1.eq(targets).sum().item()
            correct_model2 += pred2.eq(targets).sum().item()
            correct_ensemble += pred_ensemble.eq(targets).sum().item()

            total_samples += targets.size(0)

        results = {
            'model1_acc': 100.0 * correct_model1 / total_samples,
            'model2_acc': 100.0 * correct_model2 / total_samples,
            'ensemble_acc': 100.0 * correct_ensemble / total_samples,
            'samples': total_samples
        }

        return results

    def evaluate_clean_cifar10(self):
        """
        Evaluate on clean CIFAR-10 test set.

        Returns:
            Results dictionary
        """
        print("\nEvaluating on clean CIFAR-10 test set...")
        cifar_subset = CIFAR10Subset(selected_classes=self.selected_classes, download=False)
        test_loader = cifar_subset.get_test_loader(batch_size=64)

        results = self.evaluate_loader(test_loader)
        return results

    def evaluate_cifar10c(self, cifar10c_root='./data/CIFAR-10-C', severity=5,
                          corruption_types=None):
        """
        Evaluate on CIFAR-10-C corruptions.

        Args:
            cifar10c_root: Root directory of CIFAR-10-C dataset
            severity: Corruption severity (1-5)
            corruption_types: List of corruptions to test (None = all)

        Returns:
            Dictionary mapping corruption -> results
        """
        print(f"\nEvaluating on CIFAR-10-C (severity={severity})...")

        # Load CIFAR-10-C
        cifar10c = CIFAR10CCorrupted(
            root=cifar10c_root,
            selected_classes=self.selected_classes,
            corruption_types=corruption_types,
            severity=severity
        )

        # Evaluate on each corruption
        all_results = {}

        for corruption in cifar10c.corruption_types:
            loader = cifar10c.load_corruption(corruption)
            if loader is None:
                continue

            results = self.evaluate_loader(loader)
            all_results[corruption] = results

            print(f"  {corruption}: Ensemble={results['ensemble_acc']:.2f}%")

        return all_results

    def print_summary(self, clean_results, corruption_results):
        """
        Print a formatted summary of evaluation results.

        Args:
            clean_results: Results on clean CIFAR-10
            corruption_results: Results on CIFAR-10-C corruptions
        """
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Clean CIFAR-10 results
        if clean_results:
            print("\nClean CIFAR-10 Test Set:")
            print(f"  Model 1:  {clean_results['model1_acc']:.2f}%")
            print(f"  Model 2:  {clean_results['model2_acc']:.2f}%")
            print(f"  Ensemble: {clean_results['ensemble_acc']:.2f}%")

        # CIFAR-10-C results
        if corruption_results:
            print("\n" + "-"*80)
            print("CIFAR-10-C Corruption Results:")
            print("-"*80)

            # Prepare table data
            table_data = []
            for corruption, results in sorted(corruption_results.items()):
                table_data.append([
                    corruption,
                    f"{results['model1_acc']:.2f}%",
                    f"{results['model2_acc']:.2f}%",
                    f"{results['ensemble_acc']:.2f}%"
                ])

            # Print table
            headers = ["Corruption", "Model 1", "Model 2", "Ensemble"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Average results
            avg_model1 = np.mean([r['model1_acc'] for r in corruption_results.values()])
            avg_model2 = np.mean([r['model2_acc'] for r in corruption_results.values()])
            avg_ensemble = np.mean([r['ensemble_acc'] for r in corruption_results.values()])

            print("\nAverage across all corruptions:")
            print(f"  Model 1:  {avg_model1:.2f}%")
            print(f"  Model 2:  {avg_model2:.2f}%")
            print(f"  Ensemble: {avg_ensemble:.2f}%")
            print(f"\n  Ensemble improvement over Model 1: {avg_ensemble - avg_model1:+.2f}%")
            print(f"  Ensemble improvement over Model 2: {avg_ensemble - avg_model2:+.2f}%")

        print("="*80)


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate dual model on CIFAR-10-C')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--cifar10c-root', type=str, default='./data/CIFAR-10-C',
                        help='Root directory of CIFAR-10-C dataset')
    parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help='Corruption severity level (1-5)')
    parser.add_argument('--skip-clean', action='store_true',
                        help='Skip evaluation on clean CIFAR-10')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Specific corruptions to test (default: all)')

    args = parser.parse_args()

    # Configuration
    DEVICE = 'cpu'

    print("="*80)
    print("Dual Model Evaluation - CIFAR-10-C Robustness Testing")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Severity: {args.severity}")
    print("="*80)

    # Initialize evaluator (auto-detects classes from checkpoint)
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        selected_classes=None,  # Auto-detect from checkpoint
        device=DEVICE
    )

    # Get class names for display
    class_names = get_cifar10_class_names()
    selected_names = [class_names[i] for i in evaluator.selected_classes]
    print(f"Classes being evaluated: {selected_names}")

    # Evaluate on clean CIFAR-10
    clean_results = None
    if not args.skip_clean:
        clean_results = evaluator.evaluate_clean_cifar10()

    # Evaluate on CIFAR-10-C
    corruption_results = evaluator.evaluate_cifar10c(
        cifar10c_root=args.cifar10c_root,
        severity=args.severity,
        corruption_types=args.corruptions
    )

    # Print summary
    evaluator.print_summary(clean_results, corruption_results)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
