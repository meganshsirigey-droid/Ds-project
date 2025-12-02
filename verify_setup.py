#!/usr/bin/env python3
"""
Quick verification script to test if all modules can be imported and scripts are executable.
Run this before executing the main training scripts.
"""
import sys
import subprocess

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 70)
    print("VERIFYING MODULE IMPORTS")
    print("=" * 70)
    
    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'NumPy'),
        ('tqdm', 'tqdm'),
        ('sklearn', 'scikit-learn'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} import failed: {e}")
            all_ok = False
    
    print()
    
    # Test project modules
    print("VERIFYING PROJECT MODULES")
    print("=" * 70)
    
    project_modules = [
        ('models.mnist_cnn', 'SimpleCNN model'),
        ('utils.train_eval', 'Training utilities'),
        ('utils.dataloader_adapter', 'Data loader adapter'),
        ('federated.fed_utils', 'FedAvg utilities'),
        ('federated.robust_fed_utils', 'Robust federated utilities'),
    ]
    
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} import failed: {e}")
            all_ok = False
    
    return all_ok

def test_scripts():
    """Test if all main scripts can be executed (help command)."""
    print()
    print("=" * 70)
    print("VERIFYING SCRIPT EXECUTABILITY")
    print("=" * 70)
    
    scripts = [
        'level1_main.py',
        'level2_main.py',
        'level3_main.py',
    ]
    
    all_ok = True
    for script in scripts:
        try:
            result = subprocess.run(
                [sys.executable, script, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✓ {script} is executable")
            else:
                print(f"✗ {script} failed with return code {result.returncode}")
                print(f"  Error: {result.stderr}")
                all_ok = False
        except Exception as e:
            print(f"✗ {script} failed: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("FEDGUARD-MNIST SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    imports_ok = test_imports()
    scripts_ok = test_scripts()
    
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if imports_ok and scripts_ok:
        print("✓ All checks passed! The project is ready to run.")
        print()
        print("You can now execute:")
        print("  python3 level1_main.py --epochs 30 --batch_size 128 --lr 0.001")
        print("  python3 level2_main.py --rounds 30 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001")
        print("  python3 level3_main.py --rounds 10 --num_clients 10 --local_epochs 2 --batch_size 128 --lr 0.001 \\")
        print("      --malicious_client 0 --malicious_start_round 4 --attack_type sign_flip --attack_scale 5.0 \\")
        print("      --detection_threshold 0.5")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above before running training scripts.")
        return 1

if __name__ == '__main__':
    sys.exit(main())



