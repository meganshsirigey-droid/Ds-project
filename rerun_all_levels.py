#!/usr/bin/env python3
"""
Script to re-execute all three levels with full training parameters.
This will overwrite existing checkpoints with new models.
"""
import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(cmd, description, timeout=None):
    """Run a command and show real-time output"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Print output in real-time
    output_lines = []
    try:
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
        process.wait()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        process.terminate()
        return None, None, -1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"Completed in {duration:.2f}s ({duration/60:.2f} minutes)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return ''.join(output_lines), None, process.returncode

def main():
    print("="*70)
    print("FEDGUARD-MNIST: Re-Executing All Three Levels")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  WARNING: This will overwrite existing checkpoints!")
    print("Press Ctrl+C within 5 seconds to cancel...")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 1
    
    results = []
    
    # Level 1 - Full training (30 epochs)
    print("\n" + "="*70)
    print("LEVEL 1: CENTRALIZED LEARNING")
    print("="*70)
    stdout, stderr, code = run_command(
        ['python3', 'level1_main.py', 
         '--epochs', '30', 
         '--batch_size', '128', 
         '--lr', '0.001'],
        'Level 1: Centralized Training (30 epochs)'
    )
    
    if code == 0:
        print("✓ Level 1 completed successfully!")
        results.append(('Level 1', 'SUCCESS'))
    else:
        print(f"✗ Level 1 failed with exit code {code}")
        results.append(('Level 1', 'FAILED'))
        return 1
    
    # Level 2 - Full training (30 rounds)
    print("\n" + "="*70)
    print("LEVEL 2: FEDERATED LEARNING (FedAvg)")
    print("="*70)
    stdout, stderr, code = run_command(
        ['python3', 'level2_main.py', 
         '--rounds', '30', 
         '--num_clients', '10', 
         '--local_epochs', '2',
         '--batch_size', '128', 
         '--lr', '0.001'],
        'Level 2: Federated Training (30 rounds, 2 local epochs)'
    )
    
    if code == 0:
        print("✓ Level 2 completed successfully!")
        results.append(('Level 2', 'SUCCESS'))
    else:
        print(f"✗ Level 2 failed with exit code {code}")
        results.append(('Level 2', 'FAILED'))
        return 1
    
    # Level 3 - Full training (10 rounds)
    print("\n" + "="*70)
    print("LEVEL 3: ROBUST FEDERATED LEARNING WITH ATTACK DETECTION")
    print("="*70)
    stdout, stderr, code = run_command(
        ['python3', 'level3_main.py', 
         '--rounds', '10', 
         '--num_clients', '10', 
         '--local_epochs', '2',
         '--batch_size', '128', 
         '--lr', '0.001',
         '--malicious_client', '0', 
         '--malicious_start_round', '4', 
         '--attack_type', 'sign_flip', 
         '--attack_scale', '5.0',
         '--detection_threshold', '0.5'],
        'Level 3: Robust Federated Training (10 rounds, attack from round 4)'
    )
    
    if code == 0:
        print("✓ Level 3 completed successfully!")
        results.append(('Level 3', 'SUCCESS'))
    else:
        print(f"✗ Level 3 failed with exit code {code}")
        results.append(('Level 3', 'FAILED'))
        return 1
    
    # Final Summary
    print("\n" + "="*70)
    print("RE-EXECUTION SUMMARY")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for level, status in results:
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {level}: {status}")
    
    print("\n" + "="*70)
    print("All checkpoints have been updated in the 'checkpoints/' directory:")
    print("  - checkpoints/level1_best_model.pth")
    print("  - checkpoints/level2_global_best_model.pth")
    print("  - checkpoints/level3_robust_best_model.pth")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user!")
        sys.exit(1)



