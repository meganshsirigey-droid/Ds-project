# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python test_imports.py
   ```

3. **Add instructor dataloaders (if you have them):**
   - Place `dataloader4level1.py` in the project root
   - Place federated dataloader (if provided) in the project root
   - The code will automatically detect and use them

## Running the Project

### Level 1 - Centralized Learning
```bash
# Train
python level1_main.py --epochs 20

# Test saved model
python level1_main.py --test_only
```

### Level 2 - Federated Learning
```bash
# Train
python level2_main.py --rounds 20

# Test saved model
python level2_main.py --test_only
```

### Level 3 - Robust Federated Learning (For Demo Day)
```bash
# Train (10 rounds as required)
python level3_main.py --rounds 10 --malicious_start_round 4

# Test saved model
python level3_main.py --test_only
```

## Key Features Implemented

✅ **Level 1:** Centralized CNN training with model saving/loading  
✅ **Level 2:** FedAvg with 10 IID clients  
✅ **Level 3:** Trust-weighted aggregation with:
   - Cosine similarity for update consistency
   - Local Outlier Factor (LOF) for anomaly detection
   - Malicious client detection and exclusion
   - Attack simulation (sign_flip, random, scale, zero)

## Model Checkpoints

Saved models will be in `checkpoints/`:
- `level1_best_model.pth`
- `level2_global_best_model.pth`
- `level3_robust_best_model.pth`

## For Demo Day

The TA will run Level 3 with:
- 10 rounds (already default)
- Their own test dataset
- Pre-trained model checkpoints

Make sure to:
1. Train Level 3 and save the best model
2. Test that loading the model works: `python level3_main.py --test_only`
3. Ensure the code runs without errors

## Troubleshooting

- **Import errors:** Run `python test_imports.py` to check dependencies
- **CUDA errors:** Code automatically falls back to CPU
- **Dataloader errors:** Code falls back to standard MNIST if instructor dataloader not found

