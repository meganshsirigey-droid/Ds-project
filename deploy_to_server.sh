#!/bin/bash
# Script to deploy and run the project on TA's server
# Usage: ./deploy_to_server.sh

SERVER="10.51.197.111"
PORT="2223"
USER="Meghansh Siregey"
REMOTE_DIR="~/DSproject"
LOCAL_DIR="/Users/meghanshsirigey/Desktop/DSproject"

echo "=========================================="
echo "Deploying FedGuard-MNIST to TA's Server"
echo "=========================================="
echo ""
echo "Server: $USER@$SERVER:$PORT"
echo ""

# Step 1: Create deployment package
echo "Step 1: Creating deployment package..."
cd "$LOCAL_DIR"
tar -czf DSproject_deploy.tar.gz \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    --exclude='data' \
    --exclude='checkpoints' \
    --exclude='*.pyc' \
    --exclude='venv' \
    *.py *.md *.txt models/ utils/ federated/ 2>/dev/null

if [ -f "DSproject_deploy.tar.gz" ]; then
    echo "✓ Package created: DSproject_deploy.tar.gz"
    ls -lh DSproject_deploy.tar.gz
else
    echo "✗ Failed to create package"
    exit 1
fi

echo ""
echo "Step 2: Transferring files to server..."
echo "You will be prompted for your password: Meghansh0801!"
echo ""

# Transfer the package
scp -P $PORT DSproject_deploy.tar.gz "$USER@$SERVER:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo "✓ Files transferred successfully"
else
    echo "✗ Transfer failed"
    exit 1
fi

echo ""
echo "Step 3: Setting up on server..."
echo "Connecting to server to extract and setup..."
echo ""

# SSH commands to run on server
ssh -p $PORT "$USER@$SERVER" << 'ENDSSH'
cd ~/DSproject
echo "Extracting files..."
tar -xzf DSproject_deploy.tar.gz
rm DSproject_deploy.tar.gz

echo "Checking Python..."
python3 --version

echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Verifying setup..."
python3 verify_setup.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001"
echo "  python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001"
echo "  python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \\"
echo "      --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \\"
echo "      --detection_threshold 0.5"
ENDSSH

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="



