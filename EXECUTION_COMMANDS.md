# Execution Commands for Professor/TA Testing

This document contains the exact commands to execute all three levels of the FedGuard-MNIST project.

## Prerequisites

```bash
# Navigate to project directory
cd /path/to/DSproject

# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify setup (optional but recommended)
python3 verify_setup.py
```

## Level 1 - Centralized Learning

**Command:**
```bash
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001
```

**What it does:**
- Trains SimpleCNN on MNIST for 30 epochs
- Automatically saves best model to `checkpoints/level1_best_model.pth`
- Expected accuracy: ~99.60%
- Training time: ~35-40 minutes

**Test saved model:**
```bash
python3 level1_main.py --test_only --save_path checkpoints/level1_best_model.pth
```

## Level 2 - Federated Learning (FedAvg)

**Command:**
```bash
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001
```

**What it does:**
- Trains federated model with 10 IID clients using FedAvg
- 30 communication rounds, 2 local epochs per client
- Automatically saves best model to `checkpoints/level2_global_best_model.pth`
- Expected accuracy: ~98.80%
- Training time: ~85-95 minutes

**Test saved model:**
```bash
python3 level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth
```

## Level 3 - Robust Federated Learning with Attack Detection

**Command:**
```bash
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5
```

**What it does:**
- Trains robust federated model with attack detection
- 10 communication rounds, 2 local epochs per client
- Malicious client (client 0) starts attacking from round 4
- Attack type: sign-flip with scale 5.0
- Automatically saves best model to `checkpoints/level3_robust_best_model.pth`
- Expected accuracy: ~97.66%
- Expected detection rate: 100% (7/7 attack rounds detected)
- Training time: ~25-30 minutes

**Test saved model:**
```bash
python3 level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

## Quick Verification (No Training)

To verify all scripts work without running full training:

```bash
# Check help/arguments for each script
python3 level1_main.py --help
python3 level2_main.py --help
python3 level3_main.py --help

# Or run comprehensive verification
python3 verify_setup.py
```

## Expected Output Locations

All models are automatically saved to the `checkpoints/` directory:
- `checkpoints/level1_best_model.pth` - Level 1 best model
- `checkpoints/level2_global_best_model.pth` - Level 2 best model
- `checkpoints/level3_robust_best_model.pth` - Level 3 best model

## Re-Executing All Levels

**To re-run all three levels from scratch (overwrites existing checkpoints):**

**Option 1: Automated Script (Recommended)**
```bash
python3 rerun_all_levels.py
```

This will:
- Re-execute all three levels with full training parameters
- Show real-time progress
- Overwrite existing checkpoints
- Provide completion summary

**Option 2: Individual Commands**
Simply run the commands above again. They will overwrite existing checkpoints automatically.

**Note:** Re-executing is safe and will create fresh models. Previous checkpoints will be replaced.

## Notes

1. **First Run:** MNIST dataset will be automatically downloaded (~60MB) on first execution
2. **Device:** Code automatically uses GPU if available, otherwise falls back to CPU
3. **Progress:** All scripts show detailed progress bars and logging during execution
4. **Best Model:** Models are saved based on validation accuracy (Level 1) or test accuracy (Levels 2 & 3)
5. **Re-Execution:** You can re-run any level at any time - it will overwrite the checkpoint with a new model

## Troubleshooting

If you encounter import errors:
```bash
# Verify all dependencies
pip install -r requirements.txt

# Or manually install
pip install torch torchvision numpy tqdm scikit-learn
```

If scripts don't execute:
```bash
# Check Python version (should be 3.8-3.11)
python3 --version

# Verify scripts are executable
chmod +x level1_main.py level2_main.py level3_main.py
```

