# How to Run FedGuard-MNIST Project

## Quick Start Commands

### Level 1: Centralized Learning
```bash
# Train the model (default: 20 epochs)
python3 level1_main.py

# Train with custom settings
python3 level1_main.py --epochs 20 --batch_size 128 --lr 0.001

# Test only (load saved model)
python3 level1_main.py --test_only
```

### Level 2: Federated Learning (FedAvg)
```bash
# Train the federated model (default: 20 rounds, 10 clients)
python3 level2_main.py

# Train with custom settings
python3 level2_main.py --rounds 20 --num_clients 10 --local_epochs 1

# Test only (load saved model)
python3 level2_main.py --test_only
```

### Level 3: Robust Federated Learning with Attack Detection
```bash
# Train with attack detection (default: 10 rounds, attack from round 4)
python3 level3_main.py

# Train with custom attack settings
python3 level3_main.py --rounds 10 --malicious_start_round 4 --attack_type sign_flip

# Test only (load saved model)
python3 level3_main.py --test_only
```

## What You'll See

The code now shows **clear, real-time progress** with:

1. **Setup Phase**: Device info, data loading progress, model initialization
2. **Training Phase**: 
   - Per-epoch progress with timestamps
   - Training and validation metrics
   - Learning rate updates
   - Best model saves
3. **Evaluation Phase**: Final test results with clear formatting

## Example Output

```
======================================================================
LEVEL 1: CENTRALIZED LEARNING FOR MNIST CLASSIFICATION
======================================================================
Start Time: 2025-01-XX XX:XX:XX
Configuration:
  - Epochs: 20
  - Batch Size: 128
  - Learning Rate: 0.001
======================================================================

[SETUP] Using device: cpu

[DATA] Loading MNIST dataset...
[DATA] ✓ Dataset loaded in 2.34s
  - Training samples: 48,000
  - Validation samples: 12,000
  - Test samples: 10,000

[MODEL] Initializing CNN model...
[MODEL] ✓ Model initialized
  - Total parameters: 1,234,567
  - Trainable parameters: 1,234,567

[OPTIMIZER] Adam optimizer with LR scheduler initialized

======================================================================
[TRAINING] Starting training for 20 epochs...
======================================================================

──────────────────────────────────────────────────────────────────────
[EPOCH 1/20] 14:30:15
──────────────────────────────────────────────────────────────────────
[TRAIN] Training on 48,000 samples...
Training: 100%|██████████| 375/375 [01:07<00:00,  5.89it/s]
[VALID] Validating on 12,000 samples...
Evaluating: 100%|██████████| 94/94 [00:03<00:00, 29.95it/s]

[RESULTS] Epoch 1 Summary:
  Train Loss: 0.2463  |  Train Acc: 92.37%
  Val Loss:   0.0870  |  Val Acc:   97.29%
  Learning Rate: 0.001000
  Time: 70.45s
  ✓ NEW BEST MODEL! Validation Accuracy: 97.29%
```

## For Demo Day (Level 3)

Run this command exactly as the TA will:
```bash
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 1 --batch_size 128 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip \
    --attack_scale 5.0 --detection_threshold 0.5
```

This will:
- Train for 10 rounds (as required)
- Show clear progress for each round
- Display attack detection statistics
- Save the best model automatically

## Troubleshooting

1. **Import errors**: Run `python3 test_imports.py` to verify dependencies
2. **CUDA errors**: Code automatically uses CPU if CUDA unavailable
3. **Slow training**: Reduce batch size or epochs for faster testing

