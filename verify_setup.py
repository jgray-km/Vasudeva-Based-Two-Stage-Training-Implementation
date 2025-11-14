"""
Verification script to check that everything is set up correctly
Run this before starting training to catch any issues
"""

import sys
import os

# Fix for Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_imports():
    """Check that all required packages are installed."""
    print("Checking Python packages...")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'tabulate': 'tabulate',
        'matplotlib': 'matplotlib'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All packages installed!")
        return True


def check_python_version():
    """Check Python version."""
    print("\nChecking Python version...")
    version = sys.version_info

    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ⚠ Python {version.major}.{version.minor}.{version.micro}")
        print("  Recommended: Python 3.8 or higher")
        return True  # Don't fail, just warn


def check_files():
    """Check that all required files exist."""
    print("\nChecking project files...")

    required_files = [
        'model.py',
        'dataset.py',
        'train.py',
        'evaluate.py',
        'utils.py',
        'requirements.txt'
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            missing.append(file)

    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("✓ All project files present!")
        return True


def check_model_architecture():
    """Test that the model architecture works."""
    print("\nTesting model architecture...")

    try:
        import torch
        from model import SimpleCNN, DualModelFramework, compute_mi_penalty

        # Test single model
        model = SimpleCNN(num_classes=4, representation_dim=64)
        x = torch.randn(2, 3, 32, 32)
        logits, z = model(x, return_representation=True)

        assert logits.shape == (2, 4), "Single model output shape incorrect"
        assert z.shape == (2, 64), "Representation shape incorrect"
        print("  ✓ SimpleCNN works")

        # Test dual model
        dual_model = DualModelFramework(num_classes=4, representation_dim=64)
        logits1, logits2, z1, z2 = dual_model(x, return_representations=True)

        assert logits1.shape == (2, 4), "Model 1 output shape incorrect"
        assert logits2.shape == (2, 4), "Model 2 output shape incorrect"
        assert z1.shape == (2, 64), "Model 1 representation shape incorrect"
        assert z2.shape == (2, 64), "Model 2 representation shape incorrect"
        print("  ✓ DualModelFramework works")

        # Test MI penalty
        mi_penalty = compute_mi_penalty(z1, z2)
        assert mi_penalty.item() >= 0, "MI penalty should be non-negative"
        print("  ✓ MI penalty computation works")

        print("✓ Model architecture verified!")
        return True

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False


def check_dataset_loading():
    """Test that dataset loading works (without downloading)."""
    print("\nChecking dataset utilities...")

    try:
        from dataset import CIFAR10Subset, get_cifar10_class_names

        # Check class names
        class_names = get_cifar10_class_names()
        assert len(class_names) == 10, "Should have 10 class names"
        print(f"  ✓ CIFAR-10 classes: {class_names[:4]}")

        print("✓ Dataset utilities work!")
        print("  Note: CIFAR-10 will be downloaded on first training run (~170MB)")
        return True

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False


def check_directories():
    """Check/create necessary directories."""
    print("\nChecking directories...")

    directories = ['checkpoints', 'data']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✓ Created {directory}/")
        else:
            print(f"  ✓ {directory}/ exists")

    return True


def check_cuda():
    """Check if CUDA is available (optional)."""
    print("\nChecking device availability...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print("  Note: This implementation is designed for CPU, but CUDA can be used")
        else:
            print("  ✓ CPU only (expected for this project)")

        return True

    except Exception as e:
        print(f"  ⚠ Could not check CUDA: {str(e)}")
        return True  # Not critical


def print_summary(results):
    """Print final summary."""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    print("="*70)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for usage instructions")
        print("  2. Run: python train.py")
        print("  3. Wait for training to complete (~30 min - 2 hours)")
        print("  4. Run: python evaluate.py --checkpoint checkpoints/best_model.pth")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("   Install missing packages: pip install -r requirements.txt")

    return all_passed


def main():
    """Run all verification checks."""
    print("="*70)
    print("SETUP VERIFICATION")
    print("="*70)
    print("This script checks that your environment is ready for training.\n")

    results = {}

    # Run checks
    results["Python Version"] = check_python_version()
    results["Required Packages"] = check_imports()
    results["Project Files"] = check_files()
    results["Directories"] = check_directories()
    results["Model Architecture"] = check_model_architecture()
    results["Dataset Utilities"] = check_dataset_loading()
    results["Device Check"] = check_cuda()

    # Print summary
    success = print_summary(results)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
