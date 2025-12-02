# Setup Instructions for TA's Server

## Connection Details
- **Hostname:** 10.51.197.111
- **Port:** 2223
- **Username:** Meghansh Siregey

## Step 1: Connect to Server

```bash
ssh -p 2223 "Meghansh Siregey"@10.51.197.111
# Enter password when prompted
```

## Step 2: Transfer Project Files

**From your local machine, transfer the entire project:**

```bash
# From your local machine (in DSproject directory)
cd /Users/meghanshsirigey/Desktop/DSproject

# Transfer entire project directory
scp -r -P 2223 . "Meghansh Siregey"@10.51.197.111:~/DSproject/
```

**Or use rsync (better for updates):**
```bash
rsync -avz -e "ssh -p 2223" . "Meghansh Siregey"@10.51.197.111:~/DSproject/
```

## Step 3: On the Server - Setup Environment

Once connected to the server:

```bash
# Navigate to project directory
cd ~/DSproject

# Check Python version (should be 3.8-3.11)
python3 --version

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python3 verify_setup.py
```

## Step 4: Run the Code

**Level 1 - Centralized Learning:**
```bash
python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001
```

**Level 2 - Federated Learning:**
```bash
python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001
```

**Level 3 - Robust Federated Learning:**
```bash
python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \
    --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \
    --detection_threshold 0.5
```

## Step 5: Test Saved Models

After training, test the saved models:

```bash
# Test Level 1
python3 level1_main.py --test_only --save_path checkpoints/level1_best_model.pth

# Test Level 2
python3 level2_main.py --test_only --save_path checkpoints/level2_global_best_model.pth

# Test Level 3
python3 level3_main.py --test_only --save_path checkpoints/level3_robust_best_model.pth
```

## Troubleshooting

**If connection fails:**
- Check if you need VPN access
- Verify network connectivity
- Check firewall settings

**If Python/pip issues:**
- Try `python` instead of `python3`
- Use `pip3` instead of `pip`
- Check if virtual environment is activated

**If import errors:**
- Run `python3 verify_setup.py` to check dependencies
- Reinstall: `pip install -r requirements.txt --force-reinstall`

**If CUDA/GPU issues:**
- Code automatically falls back to CPU if GPU unavailable
- Check GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`

## Expected Results

- **Level 1:** ~99.60% accuracy
- **Level 2:** ~98.80% accuracy  
- **Level 3:** ~97.66% accuracy, 100% attack detection rate

## Files to Transfer

Make sure these files/directories are transferred:
- All `.py` files in root directory
- `models/` directory
- `utils/` directory
- `federated/` directory
- `requirements.txt`
- `README.md`
- `verify_setup.py`
- `EXECUTION_COMMANDS.md`

**Note:** `checkpoints/` and `data/` directories will be created automatically on first run.



