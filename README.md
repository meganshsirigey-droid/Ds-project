# FedGuard-MNIST: Secure and Robust Federated Learning with Malicious Client Detection

**Course:** CSC 8370 – Data Security  
**Project Level:** Level 3 – Robust Federated Learning with Attack Detection  
**Authors:** Nanditha Kavuri (002858183), Meghansh Siregey (002894587)

---

## Overview

This repository implements a three-level project on MNIST classification using PyTorch:

- **Level 1 (Centralized):** Standard centralized training on MNIST dataset
- **Level 2 (Federated):** Federated learning with 10 IID clients using FedAvg
- **Level 3 (Robust Federated):** Federated learning with malicious client detection using trust-weighted aggregation with cosine similarity and Local Outlier Factor (LOF)

The implementation follows the project proposal and uses trust-weighted aggregation mechanism as specified.

---

## Quick Start - Execution Commands

**For Professor/TA Testing:** Use these exact commands to run all three levels.

**To Re-Execute All Levels:** See the [Re-Execution](#re-executing-all-levels) section below.

### Prerequisites
```bash
# Navigate to project directory
cd /path/to/DSproject

# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify dependencies are installed
python3 -c "import torch; import torchvision; import numpy; import tqdm; import sklearn; print('All dependencies OK')"
```

### Level 1 - Centralized Learning
```bash
# Train Level 1 (30 epochs, saves best model automatically)
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001

# Test saved model (after training)
python3 level1_main.py --test_only --save_path checkpoints/level1_best_model.pth
```

**Expected Output:** Best model saved to `checkpoints/level1_best_model.pth` with ~99.60% accuracy

### Level 2 - Federated Learning (FedAvg)
```bash
# Train Level 2 (30 rounds, 10 clients, 2 local epochs)
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001

# Test saved model (after training)
python3 level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth
```

**Expected Output:** Best model saved to `checkpoints/level2_global_best_model.pth` with ~98.80% accuracy

### Level 3 - Robust Federated Learning with Attack Detection
```bash
# Train Level 3 (10 rounds, malicious client starts at round 4)
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5

# Test saved model (after training)
python3 level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

**Expected Output:** 
- Best model saved to `checkpoints/level3_robust_best_model.pth` with ~97.66% accuracy
- Attack detection rate: 100% (7/7 attack rounds detected)
- Detection statistics printed at end of training

### All-in-One Test (Verify All Levels Work)
```bash
# Quick verification that all scripts can run (dry run with help)
python3 level1_main.py --help
python3 level2_main.py --help
python3 level3_main.py --help
```

**Note:** First run will download MNIST dataset automatically (~60MB). Training times:
- Level 1: ~35-40 minutes (30 epochs)
- Level 2: ~85-95 minutes (30 rounds)
- Level 3: ~25-30 minutes (10 rounds)

### Re-Executing All Levels

**To re-run all three levels from scratch (will overwrite existing checkpoints):**

**Option 1: Use the automated script (Recommended)**
```bash
python3 rerun_all_levels.py
```

This script will:
- Re-execute all three levels with full training parameters
- Show real-time progress for each level
- Overwrite existing checkpoints with new models
- Provide a summary at the end

**Option 2: Run each level individually**
```bash
# Re-run Level 1
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001

# Re-run Level 2
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001

# Re-run Level 3
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5
```

**Note:** Re-executing will overwrite existing checkpoint files. The best models are automatically saved during training, so previous checkpoints will be replaced with new ones.

---

## Project Structure

```
DSproject/
├── level1_main.py              # Centralized training entrypoint
├── level2_main.py              # Federated training entrypoint (FedAvg)
├── level3_main.py              # Robust federated training (malicious detection)
├── models/
│   └── mnist_cnn.py            # CNN model (SimpleCNN)
├── federated/
│   ├── fed_utils.py            # FedAvg utilities and client_update
│   └── robust_fed_utils.py     # Trust-weighted aggregation + malicious detection
├── utils/
│   ├── train_eval.py           # train_one_epoch, evaluate, save/load helpers
│   └── dataloader_adapter.py   # Adapter to use instructor dataloader or standard MNIST
├── checkpoints/                # Saved .pth model checkpoints (created at runtime)
├── dataloader4level1.py        # Placeholder for instructor's dataloader (if provided)
└── README.md                   # This file
```

---

## Requirements

### Environment Versions

- **Python:** 3.8 – 3.11 (tested on Python 3.9+)
- **PyTorch:** >= 1.9.0 (CPU or CUDA build as appropriate)
- **torchvision:** >= 0.10.0
- **numpy:** >= 1.21.0
- **tqdm:** >= 4.62.0 (for progress bars)
- **scikit-learn:** >= 1.0.0 (for LOF anomaly detection in Level 3)

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative (manual installation):**
```bash
pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.21.0 tqdm>=4.62.0 scikit-learn>=1.0.0
```

**For GPU support (NVIDIA CUDA):**
Install a CUDA-enabled PyTorch build matching your CUDA driver from [https://pytorch.org](https://pytorch.org). The code will automatically use GPU if available, otherwise falls back to CPU.

### Verify Installation

**Quick Check:**
```bash
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
```

**Comprehensive Verification:**
```bash
# Run the verification script to test all imports and scripts
python3 verify_setup.py
```

This will verify:
- All required packages are installed
- All project modules can be imported
- All main scripts are executable

---

## Dataset Instructions

### Level 1 (Centralized Learning)

If you have the instructor's `dataloader4level1.py`, place it in the project root directory. The code will automatically use it if available. Otherwise, it will fall back to standard PyTorch MNIST dataloader.

### Level 2 & 3 (Federated Learning)

If you have the instructor's federated dataloader, place it in the project root and name it appropriately (e.g., `dataloader4level2.py`). The code will automatically use it if available. Otherwise, it will create an IID partition of the MNIST dataset.

---

## How to Run (Detailed)

All commands assume you're in the project root directory and your virtual environment is activated.

**For quick execution, see the [Quick Start](#quick-start---execution-commands) section above.**

### Level 1 — Centralized Training

**Train (30 epochs - recommended for best accuracy):**
```bash
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001
```

**Train (10 epochs - faster but lower accuracy):**
```bash
python3 level1_main.py --epochs 10 --batch_size 128 --lr 0.001
```

**Test only (evaluate saved model):**
```bash
python3 level1_main.py --test_only --save_path checkpoints/level1_best_model.pth
```

**Note:** Model is automatically saved to `checkpoints/level1_best_model.pth` during training.

### Level 2 — Federated Learning (FedAvg)

**Train (30 rounds - recommended for best accuracy):**
```bash
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001
```

**Train (10 rounds - faster):**
```bash
python3 level2_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001
```

**Test saved global model:**
```bash
python3 level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth
```

**Note:** Model is automatically saved to `checkpoints/level2_global_best_model.pth` during training.

### Level 3 — Robust Federated Learning (TA grading: 10 rounds)

**Train (10 rounds; malicious client introduced from round 4):**
```bash
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5
```

**Test saved robust model:**
```bash
python3 level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

**Note:** Model is automatically saved to `checkpoints/level3_robust_best_model.pth` during training.

---

## Key Implementation Details

### Model Architecture

- **Location:** `models/mnist_cnn.py`
- **Architecture:** SimpleCNN
  - 3 convolutional blocks with batch normalization and dropout
  - 2 fully connected layers
  - Designed for MNIST (28x28 grayscale images, 10 classes)

### Level 1: Centralized Learning

- **File:** `level1_main.py`
- Standard train/validation/test loops
- Saves best model based on validation accuracy
- Uses instructor's dataloader if available, otherwise standard MNIST

### Level 2: Federated Learning (FedAvg)

- **File:** `level2_main.py`
- Partitions MNIST training dataset into 10 IID subsets
- Each client trains locally using `client_update` in `federated/fed_utils.py`
- Server aggregates with Federated Averaging (FedAvg)
- Saves best global model based on test accuracy

### Level 3: Robust Federated Learning

- **File:** `level3_main.py`
- **Defense Strategy:** Trust-Weighted Aggregation Mechanism
  - **Cosine Similarity:** Measures consistency between client updates and global model
  - **Local Outlier Factor (LOF):** Detects anomalous updates
  - **Trust Scores:** Combined metric (60% cosine similarity + 40% LOF)
  - **Weighted Aggregation:** Updates weighted by both sample size and trust score
  - **Malicious Detection:** Clients with trust score below threshold are excluded
- **Attack Simulation:**
  - Malicious client introduced from a specified round (default: round 4)
  - Attack types: `random`, `sign_flip`, `scale`, `zero`
  - Attack scale configurable
- **Detection Statistics:** Tracks detection rate and false positives

---

## Model Saving and Loading

### Automatic Best Model Saving

**All three levels automatically save the best-performing model during training:**
- **Level 1:** Saves model with best validation accuracy
- **Level 2:** Saves global model with best test accuracy
- **Level 3:** Saves robust global model with best test accuracy (under attack)

Models are saved as PyTorch checkpoints (`.pth` files) in the `checkpoints/` directory. Each checkpoint contains:
- Model state dictionary (required)
- Optional: optimizer state, epoch number, accuracy

### Loading Saved Models for Testing

**Method 1: Using `--test_only` flag (Recommended for TA testing):**

```bash
# Test Level 1 model
python level1_main.py --test_only --save_path checkpoints/level1_best_model.pth

# Test Level 2 model
python level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth

# Test Level 3 model
python level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

**Method 2: Programmatic loading (for custom testing):**

```python
from models.mnist_cnn import SimpleCNN
from utils.train_eval import load_model, evaluate
from utils.dataloader_adapter import get_level1_loaders
import torch

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model(model, "checkpoints/level1_best_model.pth", device)

_, _, test_loader = get_level1_loaders(batch_size=128)
loss, acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {acc:.2f}%")
```

### Resuming Training

To resume training from a checkpoint, modify the code to load the model and optimizer state before the training loop. The saved checkpoints include all necessary information for resuming.

---

## Expected Results

Based on our optimized implementation:

- **Level 1 (Centralized):** Test accuracy ≈ **99.60%** (30 epochs, optimized hyperparameters)
- **Level 2 (FedAvg):** Global test accuracy ≈ **98.80%** (30 rounds, 2 local epochs)
- **Level 3 (Robust):** Robust global accuracy ≈ **97.66%** under attack; malicious client detection rate **100%** (7/7 rounds detected)

**Note:** Actual results may vary slightly based on random initialization and hardware. The best-performing models are automatically saved in the `checkpoints/` directory for TA evaluation.

---

## Hyperparameters

### Tuning Options

- **Learning rate (`--lr`):** Default 1e-3, try 5e-4 to 2e-3
- **Batch size (`--batch_size`):** Default 128, try 64 or 256
- **Local epochs (`--local_epochs`):** Default 1, try 2-5 for federated
- **Global rounds (`--rounds`):** Default 20 for Level 2, 10 for Level 3
- **Detection threshold (`--detection_threshold`):** Default 0.5, adjust for sensitivity

### For Better Performance

- **Centralized:** Increase epochs to 30-40, add LR scheduler
- **Federated:** Increase global rounds or local epochs per client
- **Detection:** Adjust `--detection_threshold` in `level3_main.py` (lower = more sensitive)

---

## Troubleshooting

### Dataloader Import Errors

- Ensure `dataloader4level1.py` is present in project root (if using instructor's dataloader)
- The adapter automatically falls back to standard MNIST if instructor's dataloader is not found

### CUDA Issues

- CUDA unavailable on MacBooks is expected. Use CPU or switch to a Linux/GPU machine for faster training
- The code automatically detects and uses CPU if CUDA is not available

### Model Loading Errors

- Ensure the checkpoint file exists at the specified path
- Check that the model architecture matches (SimpleCNN)
- Verify PyTorch version compatibility

### Permission Errors

- `.pth` files are checkpoints, not executables. Load them in Python (see examples above)
- Ensure write permissions for the `checkpoints/` directory

---

## TA Testing Instructions

For Teaching Assistant evaluation, the best-performing models are automatically saved during training. To test the saved models:

### Quick Test Commands

```bash
# Ensure you're in the project root with venv activated
cd /path/to/DSproject
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Test Level 1 (Centralized)
python3 level1_main.py --test_only --save_path checkpoints/level1_best_model.pth

# Test Level 2 (Federated)
python3 level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth

# Test Level 3 (Robust Federated)
python3 level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

### Full Training Commands (If Models Not Pre-saved)

If checkpoints don't exist, run full training first:

```bash
# Train Level 1
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001

# Train Level 2
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001

# Train Level 3
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5
```

### Saved Model Locations

All best models are saved in the `checkpoints/` directory:
- `checkpoints/level1_best_model.pth` - Best centralized model (99.60% accuracy)
- `checkpoints/level2_global_best_model.pth` - Best federated model (98.80% accuracy)
- `checkpoints/level3_robust_best_model.pth` - Best robust model (97.66% accuracy, 100% detection rate)

### Model Information

Each saved checkpoint includes:
- **Model state dictionary** (required for loading)
- **Epoch/round number** (when model was saved)
- **Accuracy** (validation/test accuracy at save time)

The `--test_only` flag loads the model and evaluates it on the test set, displaying the final accuracy.

---

## Project Requirements Compliance

✅ **Level 1:** Centralized learning with neural network model  
✅ **Level 2:** Federated learning with 10 clients, FedAvg aggregation  
✅ **Level 3:** Robust federated learning with:
- Malicious client simulation (configurable start round)
- Trust-weighted aggregation with cosine similarity and LOF
- Attack detection mechanism
- Model accuracy preservation under attack
- 10 rounds training phase support

✅ **Model Saving/Loading:** 
- Implemented with `save_model()` and `load_model()` utilities in `utils/train_eval.py`
- Best-performing models automatically saved during training
- `--test_only` flag for easy model testing
- Supports resuming training from checkpoints

✅ **README.md:** Comprehensive documentation with:
- Clear instructions on running code
- Required environment versions (Python 3.8-3.11, PyTorch >=1.9.0, etc.)
- Installation steps
- Model saving/loading examples
- TA testing instructions

✅ **PyTorch Framework:** All code uses PyTorch for model design and training  
✅ **Instructor Dataloader Support:** Automatic detection and fallback to standard MNIST

---

## References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017.
2. Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML, 2018.
3. PyTorch Tutorials — Saving and Loading Models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
4. MNIST Dataset — https://huggingface.co/datasets/ylecun/mnist

---

## License

This project is for educational purposes as part of CSC 8370 Data Security course.

