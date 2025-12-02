"""
Robust Federated Learning Utilities - Trust-weighted aggregation with cosine similarity and LOF
Optimized for memory, speed, and accuracy
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils.train_eval import train_one_epoch


def client_update(model, train_loader, device, local_epochs=1, lr=0.01, optimizer=None):
    """
    Perform local training on a client's data (honest client).
    Memory-optimized: Uses state_dict copy instead of deepcopy.
    
    Args:
        model: Global model (will be copied locally)
        train_loader: Client's training data
        device: torch.device
        local_epochs: Number of local training epochs
        lr: Learning rate
        optimizer: Optional optimizer
    
    Returns:
        Updated model state dict and number of training samples
    """
    # Memory-efficient: Load global state into model instead of deepcopy
    local_model = type(model)().to(device)
    local_model.load_state_dict(model.state_dict())
    local_model.train()
    
    if optimizer is None:
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    
    criterion = nn.CrossEntropyLoss()
    for epoch in range(local_epochs):
        train_one_epoch(local_model, train_loader, device, optimizer, criterion)
    
    num_samples = len(train_loader.dataset)
    
    # Return state dict and clear model from memory
    state_dict = local_model.state_dict()
    del local_model, optimizer
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    return state_dict, num_samples


def client_update_malicious(model, train_loader, device, attack_type='sign_flip', attack_scale=5.0):
    """
    Generate malicious client update.
    
    Args:
        model: Global model
        train_loader: Client's training data (not used for malicious updates)
        device: torch.device
        attack_type: Type of attack ('random', 'sign_flip', 'scale', 'zero')
        attack_scale: Scale factor for attacks
    
    Returns:
        Malicious model state dict and fake number of samples
    """
    malicious_state = {}
    global_state = model.state_dict()
    
    if attack_type == 'random':
        # Random parameters
        for key in global_state.keys():
            malicious_state[key] = torch.randn_like(global_state[key]) * attack_scale
    
    elif attack_type == 'sign_flip':
        # Flip signs of parameters
        for key in global_state.keys():
            malicious_state[key] = -attack_scale * global_state[key]
    
    elif attack_type == 'scale':
        # Scale parameters
        for key in global_state.keys():
            malicious_state[key] = attack_scale * global_state[key]
    
    elif attack_type == 'zero':
        # Zero out parameters
        for key in global_state.keys():
            malicious_state[key] = torch.zeros_like(global_state[key])
    
    else:
        # Default: random
        for key in global_state.keys():
            malicious_state[key] = torch.randn_like(global_state[key]) * attack_scale
    
    # Fake sample count
    num_samples = len(train_loader.dataset) if train_loader else 1000
    
    return malicious_state, num_samples


def compute_cosine_similarity(update1, update2):
    """
    Compute cosine similarity between two model updates.
    Optimized with efficient vectorization.
    
    Args:
        update1: First model state dict
        update2: Second model state dict
    
    Returns:
        Cosine similarity score
    """
    # Efficiently flatten parameters into vectors
    vec1 = torch.cat([update1[key].flatten() for key in sorted(update1.keys())])
    vec2 = torch.cat([update2[key].flatten() for key in sorted(update2.keys())])
    
    # Compute cosine similarity (more efficient than F.cosine_similarity for single pair)
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)
    
    return cos_sim.item()


def compute_update_vector(state_dict, device='cpu'):
    """
    Convert model state dict to a flat vector.
    Optimized to avoid unnecessary CPU transfers.
    
    Args:
        state_dict: Model state dict
        device: Device where tensors are located
    
    Returns:
        Flattened parameter vector as numpy array
    """
    # Efficiently concatenate and convert to numpy in one go
    params = [state_dict[key].flatten().cpu() if state_dict[key].device.type != 'cpu' 
              else state_dict[key].flatten() for key in sorted(state_dict.keys())]
    return torch.cat(params).numpy()


def robust_aggregate(global_model, client_updates, client_samples, detection_threshold=0.5):
    """
    Robust aggregation using trust-weighted mechanism with cosine similarity and LOF.
    
    This implements the trust-weighted aggregation as described in the proposal:
    - Uses cosine similarity to measure update consistency
    - Uses Local Outlier Factor (LOF) for anomaly detection
    - Computes trust scores and weights updates accordingly
    
    Args:
        global_model: Global model
        client_updates: List of client model state dicts
        client_samples: List of number of samples per client
        detection_threshold: Threshold for trust score (0-1)
    
    Returns:
        Aggregated model state dict, list of detected malicious clients
    """
    num_clients = len(client_updates)
    
    if num_clients == 0:
        return global_model.state_dict(), []
    
    # Step 1: Compute cosine similarities between each update and the global model
    # Optimized: Batch compute similarities
    global_state = global_model.state_dict()
    cosine_similarities = np.array([
        compute_cosine_similarity(global_state, client_update) 
        for client_update in client_updates
    ])
    
    # Step 2: Convert updates to vectors for LOF analysis (batch processing)
    # Get device from first tensor to optimize transfers
    sample_key = next(iter(global_state.keys()))
    device = global_state[sample_key].device
    
    update_vectors = np.array([
        compute_update_vector(client_update, device) 
        for client_update in client_updates
    ])
    
    # Step 3: Apply Local Outlier Factor (LOF) for anomaly detection
    # Use n_neighbors = min(5, num_clients - 1) to avoid issues with small client counts
    n_neighbors = min(5, max(2, num_clients - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
    lof_scores = lof.fit_predict(update_vectors)
    lof_outlier_scores = -lof.negative_outlier_factor_  # Convert to positive scores
    
    # Step 4: Compute trust scores
    # Normalize cosine similarities to [0, 1]
    cos_sim_normalized = np.array(cosine_similarities)
    cos_sim_normalized = (cos_sim_normalized - cos_sim_normalized.min()) / (
        cos_sim_normalized.max() - cos_sim_normalized.min() + 1e-8
    )
    
    # Normalize LOF scores to [0, 1] (lower is better, so invert)
    lof_normalized = 1.0 / (1.0 + lof_outlier_scores)  # Invert so higher = better
    
    # Combined trust score: weighted combination of cosine similarity and LOF
    trust_scores = 0.6 * cos_sim_normalized + 0.4 * lof_normalized
    
    # Step 5: Identify malicious clients (low trust scores)
    malicious_clients = []
    for i, trust_score in enumerate(trust_scores):
        if trust_score < detection_threshold:
            malicious_clients.append(i)
    
    # Step 6: Weighted aggregation excluding or down-weighting malicious clients
    aggregated_state = {}
    
    # Calculate total samples from honest clients only
    honest_samples = sum([client_samples[i] for i in range(num_clients) if i not in malicious_clients])
    
    # If all clients are malicious, use median as fallback
    if honest_samples == 0:
        for key in global_model.state_dict().keys():
            all_updates = torch.stack([client_state[key] for client_state in client_updates])
            aggregated_state[key] = torch.median(all_updates, dim=0)[0]
        return aggregated_state, malicious_clients
    
    # Pre-compute weights for honest clients (vectorized)
    honest_weights = np.array([
        (client_samples[i] / honest_samples) * trust_scores[i] 
        if i not in malicious_clients else 0.0
        for i in range(num_clients)
    ])
    total_trust_weight = honest_weights.sum()
    
    # Vectorized aggregation
    global_state = global_model.state_dict()
    for key in global_state.keys():
        if total_trust_weight > 0:
            # Stack all updates
            stacked_updates = torch.stack([client_state[key] for client_state in client_updates])
            
            # Get device and dtype
            device = stacked_updates.device
            dtype = stacked_updates.dtype
            
            # Apply weights
            weights_tensor = torch.tensor(honest_weights, device=device, dtype=dtype)
            # Reshape for broadcasting
            while len(weights_tensor.shape) < len(stacked_updates.shape):
                weights_tensor = weights_tensor.unsqueeze(-1)
            
            aggregated_state[key] = (stacked_updates * weights_tensor).sum(dim=0) / total_trust_weight
        else:
            # Fallback: sample-weighted average
            sample_weights = np.array([
                client_samples[i] / honest_samples if i not in malicious_clients else 0.0
                for i in range(num_clients)
            ])
            stacked_updates = torch.stack([client_state[key] for client_state in client_updates])
            device = stacked_updates.device
            dtype = stacked_updates.dtype
            weights_tensor = torch.tensor(sample_weights, device=device, dtype=dtype)
            while len(weights_tensor.shape) < len(stacked_updates.shape):
                weights_tensor = weights_tensor.unsqueeze(-1)
            aggregated_state[key] = (stacked_updates * weights_tensor).sum(dim=0)
    
    return aggregated_state, malicious_clients

