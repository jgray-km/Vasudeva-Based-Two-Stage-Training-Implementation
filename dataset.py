"""
Dataset loading and preprocessing for CIFAR-10 subset and CIFAR-10-C
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os


class CIFAR10Subset:
    """
    Load a subset of CIFAR-10 classes for efficient training.
    """
    def __init__(self, root='./data', selected_classes=[0, 1, 2, 3], download=True):
        """
        Args:
            root: Root directory for dataset storage
            selected_classes: List of class indices to use (0-9 from CIFAR-10)
            download: Whether to download CIFAR-10 if not present
        """
        self.selected_classes = selected_classes
        self.num_classes = len(selected_classes)

        # Transforms for CIFAR-10
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load CIFAR-10
        print(f"Loading CIFAR-10 with classes: {selected_classes}...")
        self.trainset_full = torchvision.datasets.CIFAR10(
            root=root, train=True, download=download, transform=self.transform_train
        )

        self.testset_full = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=self.transform_test
        )

        # Filter to selected classes
        self.trainset = self._filter_dataset(self.trainset_full, train=True)
        self.testset = self._filter_dataset(self.testset_full, train=False)

        print(f"Training samples: {len(self.trainset)}")
        print(f"Test samples: {len(self.testset)}")

    def _filter_dataset(self, dataset, train=True):
        """Filter dataset to only include selected classes and remap labels."""
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in self.selected_classes:
                indices.append(idx)

        # Create subset
        subset = Subset(dataset, indices)

        # Wrap with label remapping
        return RemappedDataset(subset, self.selected_classes)

    def get_train_loader(self, batch_size=64, shuffle=True, num_workers=0):
        """Get training data loader."""
        return DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False  # CPU training
        )

    def get_test_loader(self, batch_size=64, shuffle=False, num_workers=0):
        """Get test data loader."""
        return DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False  # CPU training
        )


class RemappedDataset(Dataset):
    """
    Wrapper to remap labels from original CIFAR-10 indices to 0-based indices.
    E.g., if selected_classes = [3, 5, 7, 9], remap to [0, 1, 2, 3]
    """
    def __init__(self, subset, selected_classes):
        self.subset = subset
        self.label_map = {orig: new for new, orig in enumerate(selected_classes)}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        new_label = self.label_map[label]
        return img, new_label


class CIFAR10CCorrupted:
    """
    Load CIFAR-10-C (corrupted) dataset for OOD testing.

    Note: CIFAR-10-C must be downloaded separately from:
    https://zenodo.org/record/2535967

    The dataset should be extracted to a directory with structure:
    cifar10c_root/
        brightness.npy
        contrast.npy
        ...
        labels.npy
    """
    def __init__(self, root='./data/CIFAR-10-C', selected_classes=[0, 1, 2, 3],
                 corruption_types=None, severity=5):
        """
        Args:
            root: Root directory containing CIFAR-10-C .npy files
            selected_classes: Classes to use (must match training)
            corruption_types: List of corruption types to test (None = all available)
            severity: Corruption severity level (1-5, where 5 is most severe)
        """
        self.root = root
        self.selected_classes = selected_classes
        self.severity = severity

        # CIFAR-10-C corruption types
        self.all_corruptions = [
            'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
            'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
            'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
        ]

        if corruption_types is None:
            self.corruption_types = self.all_corruptions
        else:
            self.corruption_types = corruption_types

        # Normalization (same as CIFAR-10)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        print(f"CIFAR-10-C loader initialized for {len(self.corruption_types)} corruption types")

    def load_corruption(self, corruption_type):
        """
        Load a specific corruption type and filter to selected classes.

        Args:
            corruption_type: Name of corruption (e.g., 'gaussian_noise')

        Returns:
            DataLoader for this corruption type
        """
        if not os.path.exists(self.root):
            print(f"WARNING: CIFAR-10-C not found at {self.root}")
            print("Please download from: https://zenodo.org/record/2535967")
            return None

        corruption_path = os.path.join(self.root, f'{corruption_type}.npy')
        labels_path = os.path.join(self.root, 'labels.npy')

        if not os.path.exists(corruption_path):
            print(f"WARNING: Corruption file not found: {corruption_path}")
            return None

        # Load data (shape: [50000, 32, 32, 3])
        # Each corruption has 5 severity levels, each with 10000 images
        data = np.load(corruption_path)
        labels = np.load(labels_path)

        # Extract data for the specified severity level
        # Severity levels are stored sequentially: 0-9999 (level 1), 10000-19999 (level 2), etc.
        start_idx = (self.severity - 1) * 10000
        end_idx = self.severity * 10000

        data_severity = data[start_idx:end_idx]
        labels_severity = labels[start_idx:end_idx]

        # Filter to selected classes
        mask = np.isin(labels_severity, self.selected_classes)
        filtered_data = data_severity[mask]
        filtered_labels = labels_severity[mask]

        # Remap labels
        label_map = {orig: new for new, orig in enumerate(self.selected_classes)}
        remapped_labels = np.array([label_map[label] for label in filtered_labels])

        # Create dataset
        dataset = NumpyDataset(filtered_data, remapped_labels, self.transform)

        return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    def get_all_corruptions(self):
        """
        Get loaders for all corruption types.

        Returns:
            Dictionary mapping corruption_type -> DataLoader
        """
        loaders = {}
        for corruption in self.corruption_types:
            loader = self.load_corruption(corruption)
            if loader is not None:
                loaders[corruption] = loader
        return loaders


class NumpyDataset(Dataset):
    """Dataset wrapper for numpy arrays (used for CIFAR-10-C)."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_cifar10_class_names():
    """Return CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


if __name__ == "__main__":
    # Test dataset loading
    print("Testing CIFAR-10 subset loading...")

    # Example: Use first 4 classes (airplane, automobile, bird, cat)
    selected_classes = [0, 1, 2, 3]
    class_names = get_cifar10_class_names()
    print(f"Selected classes: {[class_names[i] for i in selected_classes]}")

    # Load dataset
    cifar_subset = CIFAR10Subset(selected_classes=selected_classes)

    # Get data loaders
    train_loader = cifar_subset.get_train_loader(batch_size=64)
    test_loader = cifar_subset.get_test_loader(batch_size=64)

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label range: {labels.min().item()} - {labels.max().item()}")

    print("\nDataset loading test passed!")

    # Test CIFAR-10-C (will warn if not downloaded)
    print("\n" + "="*50)
    print("Testing CIFAR-10-C loading...")
    cifar10c = CIFAR10CCorrupted(selected_classes=selected_classes)
    print("Note: CIFAR-10-C test will only work if you've downloaded the dataset")
