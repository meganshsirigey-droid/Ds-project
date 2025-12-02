"""
Level 3: Robust Federated Learning with Attack Detection
Implements trust-weighted aggregation with cosine similarity and LOF
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
from federated.robust_fed_utils import (
    client_update, client_update_malicious, robust_aggregate
)


def main():
    parser = argparse.ArgumentParser(
        description='Level 3: Robust Federated MNIST Training with Attack Detection'
    )
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs per client')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--malicious_client', type=int, default=0,
                       help='Client ID to make malicious (0-indexed)')
    parser.add_argument('--malicious_start_round', type=int, default=4,
                       help='Round when malicious client starts attacking')
    parser.add_argument('--attack_type', type=str, default='sign_flip',
                       choices=['random', 'sign_flip', 'scale', 'zero'],
                       help='Type of attack')
    parser.add_argument('--attack_scale', type=float, default=5.0,
                       help='Scale factor for attacks')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Trust score threshold for malicious detection')
    parser.add_argument('--save_path', type=str, default='checkpoints/level3_robust_best_model.pth',
                       help='Path to save the best model')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LEVEL 3: ROBUST FEDERATED LEARNING WITH ATTACK DETECTION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Clients: {args.num_clients}")
    print(f"  - Local Epochs: {args.local_epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Malicious Client: {args.malicious_client}")
    print(f"  - Attack Starts: Round {args.malicious_start_round}")
    print(f"  - Attack Type: {args.attack_type}")
    print(f"  - Attack Scale: {args.attack_scale}")
    print(f"  - Detection Threshold: {args.detection_threshold}")
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
    
    # Federated training with attack detection
    best_test_acc = 0.0
    detection_stats = {
        'total_rounds_with_attack': 0,
        'detected_rounds': 0,
        'false_positives': 0
    }
    training_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"[TRAINING] Starting robust federated training for {args.rounds} rounds...")
    print(f"  Malicious client: {args.malicious_client} (starts at round {args.malicious_start_round})")
    print(f"  Attack type: {args.attack_type}, Scale: {args.attack_scale}")
    print(f"  Detection threshold: {args.detection_threshold}")
    print(f"{'='*70}")
    
    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        print(f"\n{'─'*70}")
        print(f"[ROUND {round_num}/{args.rounds}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*70}")
        
        # Check if malicious client should attack
        is_attack_round = round_num >= args.malicious_start_round
        if is_attack_round:
            detection_stats['total_rounds_with_attack'] += 1
            print(f"⚠️  [ATTACK] ACTIVE: Client {args.malicious_client} is MALICIOUS")
        else:
            print(f"✓ [ATTACK] No attack (starts at round {args.malicious_start_round})")
        
        # Collect client updates
        print(f"\n[CLIENTS] Training {args.num_clients} clients...")
        client_updates = []
        client_samples = []
        
        for client_id in tqdm(range(args.num_clients), desc="  Client Training", leave=False):
            if is_attack_round and client_id == args.malicious_client:
                # Generate malicious update
                client_state, num_samples = client_update_malicious(
                    global_model,
                    client_loaders[client_id],
                    device,
                    attack_type=args.attack_type,
                    attack_scale=args.attack_scale
                )
            else:
                # Honest client update
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
        if is_attack_round:
            print(f"  ✓ {args.num_clients-1} honest clients + 1 malicious client ({total_samples:,} total samples)")
        else:
            print(f"  ✓ All {args.num_clients} clients trained ({total_samples:,} total samples)")
        
        # Robust aggregation with attack detection
        print(f"\n[DETECTION] Running robust aggregation with attack detection...")
        print(f"  - Computing cosine similarities...")
        print(f"  - Running LOF anomaly detection...")
        print(f"  - Computing trust scores...")
        det_start = time.time()
        aggregated_state, detected_malicious = robust_aggregate(
            global_model,
            client_updates,
            client_samples,
            detection_threshold=args.detection_threshold
        )
        det_time = time.time() - det_start
        
        # Update detection statistics
        detection_status = ""
        if is_attack_round:
            if args.malicious_client in detected_malicious:
                detection_stats['detected_rounds'] += 1
                detection_status = f"✅ DETECTED (Client {args.malicious_client})"
            else:
                detection_status = f"❌ NOT DETECTED (Client {args.malicious_client} missed)"
        
        # Check for false positives
        false_pos_list = []
        for detected_id in detected_malicious:
            if detected_id != args.malicious_client or not is_attack_round:
                detection_stats['false_positives'] += 1
                false_pos_list.append(detected_id)
        
        print(f"  ✓ Detection completed in {det_time:.2f}s")
        if detected_malicious:
            print(f"  - Detected clients: {detected_malicious}")
            if detection_status:
                print(f"  - {detection_status}")
            if false_pos_list:
                print(f"  - ⚠️  False positives: {false_pos_list}")
        else:
            print(f"  - No malicious clients detected")
            if is_attack_round:
                print(f"  - ⚠️  WARNING: Attack active but not detected!")
        
        global_model.load_state_dict(aggregated_state)
        
        # Evaluate on test set
        print(f"\n[EVALUATE] Evaluating global model on test set...")
        test_loss, test_acc = evaluate(global_model, test_loader, device, criterion)
        
        round_time = time.time() - round_start
        print(f"\n[RESULTS] Round {round_num} Summary:")
        print(f"  Test Loss:     {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Round Time:    {round_time:.2f}s")
        if is_attack_round:
            print(f"  Attack Status: {'DETECTED' if args.malicious_client in detected_malicious else 'NOT DETECTED'}")
        
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
    
    # Final evaluation and statistics
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
    
    # Detection statistics
    print(f"\n{'='*70}")
    print("ATTACK DETECTION STATISTICS")
    print(f"{'='*70}")
    if detection_stats['total_rounds_with_attack'] > 0:
        detection_rate = (detection_stats['detected_rounds'] / 
                         detection_stats['total_rounds_with_attack']) * 100
        print(f"  Total rounds with attack:     {detection_stats['total_rounds_with_attack']}")
        print(f"  Rounds where attack detected: {detection_stats['detected_rounds']}")
        print(f"  Detection rate:               {detection_rate:.2f}%")
        print(f"  False positives:              {detection_stats['false_positives']}")
        if detection_rate >= 90:
            print(f"  ✓ Excellent detection performance!")
        elif detection_rate >= 70:
            print(f"  ✓ Good detection performance")
        else:
            print(f"  ⚠️  Detection needs improvement")
    else:
        print("  No attacks occurred (malicious_start_round > total_rounds)")
    
    print(f"{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

