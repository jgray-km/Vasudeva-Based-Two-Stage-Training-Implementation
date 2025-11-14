"""
Compare results across different experimental runs
Useful for tracking how lambda and number of classes affect performance
"""

import torch
import os


def load_and_compare_checkpoints(checkpoint_paths, labels):
    """
    Load multiple checkpoints and compare their performance.

    Args:
        checkpoint_paths: List of paths to checkpoint files
        labels: List of labels for each checkpoint (e.g., "λ=0.5, 4 classes")
    """
    print("="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)

    results = []

    for path, label in zip(checkpoint_paths, labels):
        if not os.path.exists(path):
            print(f"\n⚠️  Checkpoint not found: {path}")
            continue

        checkpoint = torch.load(path, map_location='cpu')

        epoch = checkpoint.get('epoch', 'N/A')
        accuracy = checkpoint.get('accuracy', 'N/A')
        lambda_mi = checkpoint.get('lambda_mi', 'N/A')
        history = checkpoint.get('history', {})

        results.append({
            'label': label,
            'path': path,
            'epoch': epoch,
            'val_acc': accuracy,
            'lambda': lambda_mi,
            'final_mi_penalty': history.get('mi_penalty', [None])[-1] if history else None,
            'final_model1_acc': history.get('model1_acc', [None])[-1] if history else None,
            'final_model2_acc': history.get('model2_acc', [None])[-1] if history else None,
            'final_ensemble_acc': history.get('ensemble_acc', [None])[-1] if history else None,
        })

    # Print comparison table
    print("\n{:<25} {:<10} {:<10} {:<10} {:<12} {:<12} {:<15}".format(
        "Experiment", "Lambda", "Epoch", "MI Penalty", "Model 1", "Model 2", "Ensemble"
    ))
    print("-"*100)

    for r in results:
        mi_str = f"{r['final_mi_penalty']:.4f}" if r['final_mi_penalty'] else "N/A"
        m1_str = f"{r['final_model1_acc']:.2f}%" if r['final_model1_acc'] else "N/A"
        m2_str = f"{r['final_model2_acc']:.2f}%" if r['final_model2_acc'] else "N/A"
        ens_str = f"{r['final_ensemble_acc']:.2f}%" if r['final_ensemble_acc'] else "N/A"

        print("{:<25} {:<10} {:<10} {:<12} {:<12} {:<12} {:<15}".format(
            r['label'],
            r['lambda'],
            r['epoch'],
            mi_str,
            m1_str,
            m2_str,
            ens_str
        ))

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    for r in results:
        print(f"\n{r['label']}:")
        if r['final_model1_acc'] and r['final_model2_acc'] and r['final_ensemble_acc']:
            m1 = r['final_model1_acc']
            m2 = r['final_model2_acc']
            ens = r['final_ensemble_acc']

            diff_m1_m2 = abs(m2 - m1)
            gain_over_m1 = ens - m1
            gain_over_m2 = ens - m2
            gain_over_best = ens - max(m1, m2)

            print(f"  Model difference: {diff_m1_m2:.2f}% (higher = more diversity)")
            print(f"  Ensemble gain over M1: {gain_over_m1:+.2f}%")
            print(f"  Ensemble gain over M2: {gain_over_m2:+.2f}%")
            print(f"  Ensemble gain over best: {gain_over_best:+.2f}%")

            if r['final_mi_penalty']:
                print(f"  MI penalty: {r['final_mi_penalty']:.4f} ({'low/good' if r['final_mi_penalty'] < 0.15 else 'moderate' if r['final_mi_penalty'] < 0.25 else 'high/weak'})")

            # Diagnosis
            if gain_over_best < 0.5:
                if diff_m1_m2 < 1.0:
                    print("  ⚠️  Models too similar - increase lambda or classes")
                else:
                    print("  ⚠️  One model dominates - task may be too easy")
            elif gain_over_best < 1.5:
                print("  ✓  Modest ensemble improvement - acceptable")
            else:
                print("  ✓✓ Strong ensemble improvement - excellent!")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage - modify paths and labels based on your experiments

    checkpoints = []
    labels = []

    # Check for different experimental runs
    experiments = [
        ("checkpoints/best_model_lambda0.5.pth", "λ=0.5, 4 classes"),
        ("checkpoints/best_model_lambda1.0.pth", "λ=1.0, 4 classes"),
        ("checkpoints/best_model_lambda1.5.pth", "λ=1.5, 4 classes"),
        ("checkpoints/best_model.pth", "Latest run"),
    ]

    for path, label in experiments:
        if os.path.exists(path):
            checkpoints.append(path)
            labels.append(label)

    if not checkpoints:
        print("No checkpoints found!")
        print("\nTo use this script, save your checkpoints with descriptive names:")
        print("  - checkpoints/best_model_lambda0.5.pth")
        print("  - checkpoints/best_model_lambda1.0.pth")
        print("  - checkpoints/best_model_6classes.pth")
        print("\nThen run: python compare_experiments.py")
    else:
        load_and_compare_checkpoints(checkpoints, labels)
