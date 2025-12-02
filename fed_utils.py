"""
Federated Learning Utilities - FedAvg implementation
Optimized for memory and speed
"""
import torch
import torch.nn as nn
from utils.train_eval import train_one_epoch


def client_update(model, train_loader, device, local_epochs=1, lr=0.01, optimizer=None):
    """
    Perform local training on a client's data.
    Memory-optimized: Uses state_dict copy instead of deepcopy.
    
    Args:
        model: Global model (will be copied locally)
        train_loader: Client's training data
        device: torch.device
        local_epochs: Number of local training epochs
        lr: Learning rate
        optimizer: Optional optimizer (if None, creates SGD)
    
    Returns:
        Updated model state dict and number of training samples
    """
    # Memory-efficient: Load global state into model instead of deepcopy
    local_model = type(model)().to(device)
    local_model.load_state_dict(model.state_dict())
    local_model.train()
    
    # Setup optimizer - optimized for better accuracy
    if optimizer is None:
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    
    # Local training
    criterion = nn.CrossEntropyLoss()
    for epoch in range(local_epochs):
        train_one_epoch(local_model, train_loader, device, optimizer, criterion)
    
    # Get number of samples
    num_samples = len(train_loader.dataset)
    
    # Return state dict and clear model from memory
    state_dict = local_model.state_dict()
    del local_model, optimizer
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return state_dict, num_samples


def fedavg_aggregate(global_model, client_updates, client_samples):
    """
    Aggregate client updates using Federated Averaging (FedAvg).
    Optimized with vectorized operations.
    
    Args:
        global_model: Global model (to get parameter structure)
        client_updates: List of client model state dicts
        client_samples: List of number of samples per client
    
    Returns:
        Aggregated model state dict
    """
    # Calculate total samples and normalize weights
    total_samples = sum(client_samples)
    weights = [s / total_samples for s in client_samples]
    
    # Initialize aggregated parameters
    aggregated_state = {}
    global_state = global_model.state_dict()
    
    # Vectorized weighted average of parameters
    for key in global_state.keys():
        # Stack all client updates for this parameter
        stacked_updates = torch.stack([client_state[key] for client_state in client_updates])
        
        # Compute weighted sum efficiently
        weights_tensor = torch.tensor(weights, device=stacked_updates.device, dtype=stacked_updates.dtype)
        # Reshape weights to broadcast correctly
        while len(weights_tensor.shape) < len(stacked_updates.shape):
            weights_tensor = weights_tensor.unsqueeze(-1)
        
        aggregated_state[key] = (stacked_updates * weights_tensor).sum(dim=0)
    
    return aggregated_state

