"""
Quick test script to verify all imports work correctly.
Run this after installing dependencies to verify the setup.
"""
import sys

def test_imports():
    """Test all critical imports"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ torchvision not installed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ numpy not installed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn not installed: {e}")
        return False
    
    try:
        from models.mnist_cnn import SimpleCNN
        print("✓ Model import successful")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    try:
        from federated.fed_utils import client_update, fedavg_aggregate
        print("✓ Federated utilities import successful")
    except ImportError as e:
        print(f"✗ Federated utilities import failed: {e}")
        return False
    
    try:
        from federated.robust_fed_utils import robust_aggregate, client_update_malicious
        print("✓ Robust federated utilities import successful")
    except ImportError as e:
        print(f"✗ Robust federated utilities import failed: {e}")
        return False
    
    try:
        from utils.train_eval import train_one_epoch, evaluate, save_model, load_model
        print("✓ Training utilities import successful")
    except ImportError as e:
        print(f"✗ Training utilities import failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("All imports successful! Project is ready to use.")
    print("="*50)
    return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

