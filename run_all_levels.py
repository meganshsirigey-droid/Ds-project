"""
Script to run all three levels and generate summary results
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return result.stdout, result.stderr, result.returncode

def extract_results(output, level_name):
    """Extract key results from output"""
    results = {
        'level': level_name,
        'test_accuracy': None,
        'test_loss': None,
        'best_val_acc': None,
        'detection_rate': None
    }
    
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'Test Accuracy:' in line:
            try:
                acc = float(line.split('Test Accuracy:')[1].split('%')[0].strip())
                results['test_accuracy'] = acc
            except:
                pass
        if 'Test Loss:' in line:
            try:
                loss = float(line.split('Test Loss:')[1].strip())
                results['test_loss'] = loss
            except:
                pass
        if 'Best Validation Accuracy:' in line or 'Best Test Accuracy:' in line:
            try:
                acc = float(line.split(':')[1].split('%')[0].strip())
                results['best_val_acc'] = acc
            except:
                pass
        if 'Detection rate:' in line:
            try:
                rate = float(line.split('Detection rate:')[1].split('%')[0].strip())
                results['detection_rate'] = rate
            except:
                pass
    
    return results

def main():
    print("="*70)
    print("FEDGUARD-MNIST: Running All Three Levels")
    print("="*70)
    
    results_summary = []
    
    # Level 1 - Quick test (3 epochs)
    print("\n[LEVEL 1] Centralized Learning")
    stdout, stderr, code = run_command(
        ['python3', 'level1_main.py', '--epochs', '3', '--batch_size', '128'],
        'Level 1: Centralized Training (3 epochs)'
    )
    if code == 0:
        results = extract_results(stdout, 'Level 1')
        results_summary.append(results)
        print(f"✓ Level 1 completed: Test Accuracy = {results['test_accuracy']:.2f}%")
    else:
        print(f"✗ Level 1 failed: {stderr[:200]}")
    
    # Level 2 - Quick test (5 rounds)
    print("\n[LEVEL 2] Federated Learning (FedAvg)")
    stdout, stderr, code = run_command(
        ['python3', 'level2_main.py', '--rounds', '5', '--num_clients', '10', '--local_epochs', '1'],
        'Level 2: Federated Training (5 rounds)'
    )
    if code == 0:
        results = extract_results(stdout, 'Level 2')
        results_summary.append(results)
        print(f"✓ Level 2 completed: Test Accuracy = {results['test_accuracy']:.2f}%")
    else:
        print(f"✗ Level 2 failed: {stderr[:200]}")
    
    # Level 3 - Full test (10 rounds as required)
    print("\n[LEVEL 3] Robust Federated Learning with Attack Detection")
    stdout, stderr, code = run_command(
        ['python3', 'level3_main.py', '--rounds', '10', '--num_clients', '10', 
         '--malicious_start_round', '4', '--attack_type', 'sign_flip'],
        'Level 3: Robust Federated Training (10 rounds, attack from round 4)'
    )
    if code == 0:
        results = extract_results(stdout, 'Level 3')
        results_summary.append(results)
        print(f"✓ Level 3 completed: Test Accuracy = {results['test_accuracy']:.2f}%")
        if results['detection_rate']:
            print(f"  Detection Rate = {results['detection_rate']:.2f}%")
    else:
        print(f"✗ Level 3 failed: {stderr[:200]}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for res in results_summary:
        print(f"\n{res['level']}:")
        if res['test_accuracy']:
            print(f"  Test Accuracy: {res['test_accuracy']:.2f}%")
        if res['test_loss']:
            print(f"  Test Loss: {res['test_loss']:.4f}")
        if res['best_val_acc']:
            print(f"  Best Validation/Test Accuracy: {res['best_val_acc']:.2f}%")
        if res['detection_rate']:
            print(f"  Malicious Client Detection Rate: {res['detection_rate']:.2f}%")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

