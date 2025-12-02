"""
Level 2: Federated Learning for MNIST Classification (FedAvg)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from tqdm import tqdm
from models.mnist_cnn import SimpleCNN
from utils.train_eval import evaluate, save_model, load_model
from utils.dataloader_adapter import get_federated_loaders
from federated.fed_utils import client_update, fedavg_aggregate


def main():
    parser = argparse.ArgumentParser(description='Level 2: Federated MNIST Training (FedAvg)')
    parser.add_argument('--rounds', type=int, default=30, help='Number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs per client')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='checkpoints/level2_global_best_model.pth',
                       help='Path to save the best model')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LEVEL 2: FEDERATED LEARNING (FedAvg)")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Clients: {args.num_clients}")
    print(f"  - Local Epochs: {args.local_epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SETUP] Using device: {device}")
    
    # Load federated data
    print(f"\n[DATA] Loading federated data for {args.num_clients} clients...")
    start_time = time.time()
    client_loaders, test_loader = get_federated_loaders(
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        iid=True
    )
    load_time = time.time() - start_time
    print(f"[DATA] ✓ Dataset loaded in {load_time:.2f}s")
    print(f"  - Client dataloaders: {len(client_loaders)}")
    print(f"  - Test samples: {len(test_loader.dataset):,}")
    for i, loader in enumerate(client_loaders[:3]):  # Show first 3
        print(f"  - Client {i+1} samples: {len(loader.dataset):,}")
    if len(client_loaders) > 3:
        print(f"  - ... and {len(client_loaders)-3} more clients")
    
    # Initialize global model
    print(f"\n[MODEL] Initializing global CNN model...")
    global_model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"[MODEL] ✓ Global model initialized ({total_params:,} parameters)")
    criterion = nn.CrossEntropyLoss()
    
    if args.test_only:
        # Load and test
        load_model(global_model, args.save_path, device)
        test_loss, test_acc = evaluate(global_model, test_loader, device, criterion)
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        return
    
    # Federated training
    best_test_acc = 0.0
    training_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"[TRAINING] Starting federated training for {args.rounds} rounds...")
    print(f"  Each client trains for {args.local_epochs} local epoch(s) per round")
    print(f"{'='*70}")
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n{'─'*70}")
        print(f"[ROUND {round_num}/{args.rounds}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*70}")
        
        # Collect client updates
        print(f"[CLIENTS] Training {args.num_clients} clients locally...")
        client_updates = []
        client_samples = []
        
        for client_id in tqdm(range(args.num_clients), desc="  Client Training", leave=False):
            # Local training
            client_state, num_samples = client_update(
                global_model,
                client_loaders[client_id],
                device,
                local_epochs=args.local_epochs,
                lr=args.lr
            )
            
            client_updates.append(client_state)
            client_samples.append(num_samples)
        
        total_samples = sum(client_samples)
        print(f"  ✓ All clients trained ({total_samples:,} total samples)")
        
        # Aggregate updates (FedAvg)
        print(f"[AGGREGATE] Aggregating client updates using FedAvg...")
        agg_start = time.time()
        aggregated_state = fedavg_aggregate(global_model, client_updates, client_samples)
        global_model.load_state_dict(aggregated_state)
        agg_time = time.time() - agg_start
        print(f"  ✓ Aggregation completed in {agg_time:.2f}s")
        
        # Evaluate on test set
        print(f"[EVALUATE] Evaluating global model on test set...")
        test_loss, test_acc = evaluate(global_model, test_loader, device, criterion)
        
        round_time = time.time() - round_start
        print(f"\n[RESULTS] Round {round_num} Summary:")
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Round Time:    {round_time:.2f}s")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(global_model, args.save_path, epoch=round_num, accuracy=test_acc)
            print(f"  ✓ NEW BEST MODEL! Test Accuracy: {test_acc:.2f}%")
        else:
            improvement = test_acc - best_test_acc
            print(f"  Best so far: {best_test_acc:.2f}% (current: {improvement:+.2f}%)")
    
    total_training_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"[TRAINING] Completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"{'='*70}")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("[EVALUATION] Final Test Evaluation")
    print(f"{'='*70}")
    print("[LOAD] Loading best model checkpoint...")
    load_model(global_model, args.save_path, device)
    print(f"[TEST] Evaluating on {len(test_loader.dataset):,} test samples...")
    test_loss, test_acc = evaluate(global_model, test_loader, device, criterion)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Test Accuracy:    {test_acc:.2f}%")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

