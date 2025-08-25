# utils.py
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, class_to_forget=None):
    """
    Returns CIFAR-10 train and test data loaders.
    If class_to_forget is specified, it removes that class from the dataset.
    """
    # Transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load original datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    if class_to_forget is not None:
        print(f"Preparing dataset by removing class: {class_to_forget}...")
        
        # --- Filter training data ---
        train_indices_to_keep = [i for i, target in enumerate(trainset.targets) if target != class_to_forget]
        trainset = torch.utils.data.Subset(trainset, train_indices_to_keep)

        # --- Filter test data ---
        test_indices_to_keep = [i for i, target in enumerate(testset.targets) if target != class_to_forget]
        testset = torch.utils.data.Subset(testset, test_indices_to_keep)
        
        print(f"New training set size: {len(trainset)}")
        print(f"New test set size: {len(testset)}")

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader
