#!/usr/bin/env python3
"""
Config Test Runner
Tests all default*.json config files in the configs directory
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import json
from datetime import datetime
import argparse

def find_default_configs(configs_dir="configs"):
    """Find all default*.json files in the configs directory"""
    config_path = Path(configs_dir)
    if not config_path.exists():
        print(f"Error: {configs_dir} directory not found!")
        return []
    
    default_configs = list(config_path.glob("default*.json"))
    return sorted([config.name for config in default_configs])

def run_config_test(config_file, timeout=300, env_vars=None):
    """
    Run a single config test
    
    Args:
        config_file: Name of the config file (e.g., 'default.json')
        timeout: Maximum time to wait for completion (seconds)
        env_vars: Dictionary of environment variables to set
    
    Returns:
        dict: Test result with status, output, error, and execution time
    """
    config_path = f"configs/{config_file}"
    cmd = [sys.executable, "main.py", "--config", config_path]
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_file}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Prepare environment variables
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        print(f"Environment variables: {', '.join([f'{k}={v}' for k, v in env_vars.items()])}")
    
    try:
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            env=env
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check if the process completed successfully
        success = result.returncode == 0
        
        return {
            'config': config_file,
            'success': success,
            'returncode': result.returncode,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'timeout': False
        }
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'config': config_file,
            'success': False,
            'returncode': -1,
            'execution_time': execution_time,
            'stdout': '',
            'stderr': f'Process timed out after {timeout} seconds',
            'timeout': True
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'config': config_file,
            'success': False,
            'returncode': -1,
            'execution_time': execution_time,
            'stdout': '',
            'stderr': f'Exception occurred: {str(e)}',
            'timeout': False
        }

def print_result_summary(result):
    """Print a summary of a single test result"""
    status = "✅ PASS" if result['success'] else "❌ FAIL"
    print(f"{status} {result['config']} ({result['execution_time']:.1f}s)")
    
    if not result['success']:
        print(f"   Return code: {result['returncode']}")
        if result['timeout']:
            print(f"   Error: {result['stderr']}")
        elif result['stderr']:
            # Print first few lines of stderr
            stderr_lines = result['stderr'].strip().split('\n')
            for line in stderr_lines[:3]:  # Show first 3 error lines
                print(f"   Error: {line}")
            if len(stderr_lines) > 3:
                print(f"   ... ({len(stderr_lines) - 3} more error lines)")

def save_detailed_results(results, output_file="test_results.json"):
    """Save detailed test results to a JSON file"""
    timestamp = datetime.now().isoformat()
    
    output_data = {
        'timestamp': timestamp,
        'total_tests': len(results),
        'passed': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")

def parse_env_vars(env_string):
    """Parse environment variables from string format like 'KEY1=value1,KEY2=value2'"""
    if not env_string:
        return {}
    
    env_vars = {}
    for pair in env_string.split(','):
        if '=' in pair:
            key, value = pair.strip().split('=', 1)
            env_vars[key.strip()] = value.strip()
    return env_vars

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Test all default*.json config files')
    parser.add_argument('--env', type=str, help='Environment variables in format KEY1=value1,KEY2=value2')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per test in seconds (default: 300)')
    parser.add_argument('--gpu', type=str, help='GPU device ID (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb (sets WANDB_MODE=disabled)')
    args = parser.parse_args()
    
    print("🧪 Config Test Runner")
    print("=" * 60)
    
    # Find all default config files
    config_files = find_default_configs()
    
    if not config_files:
        print("No default*.json files found in configs directory!")
        return 1
    
    print(f"Found {len(config_files)} config files to test:")
    for config in config_files:
        print(f"  - {config}")
    
    # Ask for confirmation
    response = input(f"\nProceed with testing all {len(config_files)} configs? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Test cancelled.")
        return 0
    
    # Parse environment variables
    env_vars = parse_env_vars(args.env) if args.env else {}
    
    # Add common environment variables based on flags
    if args.gpu:
        env_vars['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.no_wandb:
        env_vars['WANDB_MODE'] = 'disabled'
    
    # Display environment variables if any
    if env_vars:
        print("\n🌍 Environment variables:")
        for key, value in env_vars.items():
            print(f"  {key}={value}")
    
    # Set timeout
    timeout = args.timeout
    
    # Run tests
    results = []
    start_time = time.time()
    
    for i, config_file in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] Testing {config_file}...")
        
        result = run_config_test(config_file, timeout, env_vars)
        results.append(result)
        
        # Print immediate result
        print_result_summary(result)
    
    # Print final summary
    total_time = time.time() - start_time
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📈 Success rate: {(passed/len(results)*100):.1f}%")
    
    # Show failed tests
    if failed > 0:
        print(f"\n❌ Failed tests:")
        for result in results:
            if not result['success']:
                print(f"  - {result['config']}: {result['stderr'][:100]}...")
    
    # Save detailed results
    save_detailed_results(results)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
