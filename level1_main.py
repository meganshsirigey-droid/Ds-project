"""
Level 1: Centralized Learning for MNIST Classification
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from models.mnist_cnn import SimpleCNN
from utils.train_eval import train_one_epoch, evaluate, save_model, load_model
from utils.dataloader_adapter import get_level1_loaders


def main():
    parser = argparse.ArgumentParser(description='Level 1: Centralized MNIST Training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='checkpoints/level1_best_model.pth',
                       help='Path to save the best model')
    parser.add_argument('--test_only', action='store_true', help='Only test the model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LEVEL 1: CENTRALIZED LEARNING FOR MNIST CLASSIFICATION")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SETUP] Using device: {device}")
    
    # Load data
    print("\n[DATA] Loading MNIST dataset...")
    start_time = time.time()
    train_loader, val_loader, test_loader = get_level1_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    load_time = time.time() - start_time
    print(f"[DATA] ✓ Dataset loaded in {load_time:.2f}s")
    print(f"  - Training samples: {len(train_loader.dataset):,}")
    print(f"  - Validation samples: {len(val_loader.dataset):,}")
    print(f"  - Test samples: {len(test_loader.dataset):,}")
    
    # Initialize model
    print("\n[MODEL] Initializing CNN model...")
    model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] ✓ Model initialized")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    # Optimized: Lower weight decay, better for high accuracy
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    
    # Learning rate scheduler for better accuracy - more aggressive
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    print(f"[OPTIMIZER] Adam optimizer with LR scheduler initialized")
    
    if args.test_only:
        # Load and test
        load_model(model, args.save_path, device)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        return
    
    # Training loop
    best_val_acc = 0.0
    training_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"[TRAINING] Starting training for {args.epochs} epochs...")
    print(f"{'='*70}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'─'*70}")
        print(f"[EPOCH {epoch}/{args.epochs}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*70}")
        
        # Train
        print(f"[TRAIN] Training on {len(train_loader.dataset):,} samples...")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )
        
        # Validate
        print(f"[VALID] Validating on {len(val_loader.dataset):,} samples...")
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[RESULTS] Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.save_path, optimizer, epoch, val_acc)
            print(f"  ✓ NEW BEST MODEL! Validation Accuracy: {val_acc:.2f}%")
        else:
            improvement = val_acc - best_val_acc
            print(f"  Best so far: {best_val_acc:.2f}% (current: {improvement:+.2f}%)")
    
    total_training_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"[TRAINING] Completed in {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"{'='*70}")
    
    # Final test evaluation
    print(f"\n{'='*70}")
    print("[EVALUATION] Final Test Evaluation")
    print(f"{'='*70}")
    print("[LOAD] Loading best model checkpoint...")
    load_model(model, args.save_path, device)
    print(f"[TEST] Evaluating on {len(test_loader.dataset):,} test samples...")
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Test Accuracy:    {test_acc:.2f}%")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

