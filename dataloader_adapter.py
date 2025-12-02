"""
Dataloader adapter to work with instructor-provided dataloaders or standard PyTorch dataloaders
"""
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import os


def get_level1_loaders(batch_size=128, data_dir='./data'):
    """
    Get dataloaders for Level 1 (Centralized Learning).
    
    First tries to import instructor's dataloader4level1.py, otherwise uses standard MNIST.
    
    Args:
        batch_size: Batch size for dataloaders
        data_dir: Directory to store/download MNIST data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Try to use instructor's dataloader
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataloader4level1 import get_dataloader
        
        # Assuming instructor's dataloader returns train, val, test
        train_loader, val_loader, test_loader = get_dataloader(batch_size)
        print("Using instructor's dataloader4level1.py")
        return train_loader, val_loader, test_loader
    except ImportError:
        print("Instructor dataloader not found, using standard MNIST dataloader")
        # Fallback to standard MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load MNIST
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
        # Split training into train and validation (80/20)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        # Optimized: Use num_workers for faster data loading, pin_memory for GPU
        num_workers = 4 if os.cpu_count() >= 4 else 2
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
        )
        
        return train_loader, val_loader, test_loader


def get_federated_loaders(num_clients=10, batch_size=128, data_dir='./data', iid=True):
    """
    Get dataloaders for Level 2 and 3 (Federated Learning).
    
    First tries to import instructor's federated dataloader, otherwise partitions MNIST.
    
    Args:
        num_clients: Number of clients
        batch_size: Batch size for each client
        data_dir: Directory to store/download MNIST data
        iid: Whether to create IID partition (True) or non-IID (False)
    
    Returns:
        List of client dataloaders, test_loader
    """
    # Try to use instructor's federated dataloader
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Assuming instructor has a federated dataloader function
        # This is a placeholder - adjust based on actual instructor dataloader
        from dataloader4level2 import get_federated_dataloader
        
        client_loaders, test_loader = get_federated_dataloader(
            num_clients=num_clients, batch_size=batch_size
        )
        print("Using instructor's federated dataloader")
        return client_loaders, test_loader
    except ImportError:
        print("Instructor federated dataloader not found, creating IID partition")
        # Fallback: Create IID partition of MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
        # Create IID partition
        if iid:
            # Randomly shuffle and split
            total_size = len(train_dataset)
            client_size = total_size // num_clients
            indices = torch.randperm(total_size).tolist()
            
            # Optimized: Use num_workers for faster data loading
            num_workers = 2 if os.cpu_count() >= 4 else 1
            pin_memory = torch.cuda.is_available()
            
            client_loaders = []
            for i in range(num_clients):
                start_idx = i * client_size
                end_idx = (i + 1) * client_size if i < num_clients - 1 else total_size
                client_indices = indices[start_idx:end_idx]
                client_subset = Subset(train_dataset, client_indices)
                client_loader = DataLoader(
                    client_subset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
                )
                client_loaders.append(client_loader)
        else:
            # Non-IID: sort by label and assign
            sorted_indices = torch.argsort(torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]))
            
            # Optimized: Use num_workers for faster data loading
            num_workers = 2 if os.cpu_count() >= 4 else 1
            pin_memory = torch.cuda.is_available()
            
            client_loaders = []
            for i in range(num_clients):
                client_indices = sorted_indices[i::num_clients].tolist()
                client_subset = Subset(train_dataset, client_indices)
                client_loader = DataLoader(
                    client_subset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
                )
                client_loaders.append(client_loader)
        
        # Optimized test loader
        num_workers = 2 if os.cpu_count() >= 4 else 1
        pin_memory = torch.cuda.is_available()
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
        )
        
        return client_loaders, test_loader

