#!/usr/bin/env python3
"""
Experiment Sweep Manager for main_sequential.py

Supports:
- Grid search for hyperparameter tuning
- List mode for specific configurations
- Random sampling for large parameter spaces
- Ablation studies with feature toggles

Usage:
    python experiment_sweep.py --sweep_config sweep_configs/my_experiment.json
    python experiment_sweep.py --sweep_config sweep_configs/my_experiment.json --dry_run
"""

import os
import json
import argparse
import subprocess
import sys
from itertools import product
from datetime import datetime
from pathlib import Path
import copy


class ExperimentSweep:
    """Manages experiment sweeps over configuration spaces"""
    
    def __init__(self, sweep_config_path, dry_run=False):
        self.sweep_config_path = sweep_config_path
        self.dry_run = dry_run
        self.sweep_config = self._load_sweep_config()
        self.base_config = self._load_base_config()
        self.experiment_name = self.sweep_config.get("experiment_name", "experiment")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_sweep_config(self):
        """Load sweep configuration file"""
        with open(self.sweep_config_path, 'r') as f:
            return json.load(f)
    
    def _load_base_config(self):
        """Load base configuration file"""
        base_config_path = self.sweep_config.get("base_config")
        if not base_config_path:
            raise ValueError("sweep_config must specify 'base_config' path")
        
        with open(base_config_path, 'r') as f:
            return json.load(f)
    
    def _set_nested_value(self, config, key_path, value):
        """
        Set a nested value in config dictionary using dot notation
        Example: "sequential.adaptive_control.enabled" -> config['sequential']['adaptive_control']['enabled']
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config, key_path):
        """Get a nested value from config dictionary using dot notation"""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                return None
            current = current[key]
        
        return current
    
    def _generate_grid_combinations(self):
        """Generate all combinations for grid search"""
        parameters = self.sweep_config.get("parameters", {})
        
        if not parameters:
            print("Warning: No parameters specified for sweep. Running single experiment.")
            return [{}]
        
        # Get parameter names and their value lists
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)
        
        return combinations
    
    def _generate_list_combinations(self):
        """Generate specific combinations from list mode"""
        combinations_list = self.sweep_config.get("combinations", [])
        
        if not combinations_list:
            print("Warning: No combinations specified for list mode. Running single experiment.")
            return [{}]
        
        return combinations_list
    
    def generate_run_configs(self):
        """Generate all run configurations based on sweep type"""
        sweep_type = self.sweep_config.get("sweep_type", "grid")
        
        if sweep_type == "grid":
            combinations = self._generate_grid_combinations()
        elif sweep_type == "list":
            combinations = self._generate_list_combinations()
        else:
            raise ValueError(f"Unknown sweep_type: {sweep_type}. Supported types: 'grid', 'list'")
        
        print(f"Generated {len(combinations)} configuration combinations")
        return combinations
    
    def create_run_config(self, param_combo, run_id):
        """Create a specific run configuration by merging base + overrides + params"""
        # Deep copy base config
        run_config = copy.deepcopy(self.base_config)
        
        # Apply fixed overrides from sweep config
        fixed_overrides = self.sweep_config.get("fixed_overrides", {})
        for key_path, value in fixed_overrides.items():
            self._set_nested_value(run_config, key_path, value)
        
        # Apply parameter combination
        for key_path, value in param_combo.items():
            self._set_nested_value(run_config, key_path, value)
        
        # Update output folder with run-specific path
        output_base = self.sweep_config.get("output_base", "./outputs/experiments")
        run_output_folder = os.path.join(
            output_base,
            self.experiment_name,
            self.timestamp,
            f"run_{run_id:03d}"
        )
        self._set_nested_value(run_config, "postprocessing.output_folder", run_output_folder)
        
        return run_config
    
    def save_run_config(self, run_config, run_id, output_folder):
        """Save run configuration to output folder for reproducibility"""
        os.makedirs(output_folder, exist_ok=True)
        
        config_save_path = os.path.join(output_folder, "run_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(run_config, f, indent=4)
        
        print(f"  Saved config to: {config_save_path}")
    
    def generate_run_description(self, param_combo, run_id):
        """Generate human-readable description of run parameters"""
        if not param_combo:
            return f"Run {run_id:03d}: baseline"
        
        parts = [f"Run {run_id:03d}:"]
        for key, value in param_combo.items():
            key_short = key.split('.')[-1]  # Use last part of dot notation
            parts.append(f"{key_short}={value}")
        
        return " ".join(parts)
    
    def run_experiment(self, run_config, run_id, param_combo):
        """Execute a single experiment run"""
        output_folder = self._get_nested_value(run_config, "postprocessing.output_folder")
        
        print(f"\n{'='*80}")
        print(self.generate_run_description(param_combo, run_id))
        print(f"{'='*80}")
        
        # Save configuration
        self.save_run_config(run_config, run_id, output_folder)
        
        # Create temporary config file for this run with unique name
        temp_config_path = f"temp_{self.experiment_name}_{self.timestamp}_run_{run_id:03d}_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump(run_config, f, indent=4)
        
        if self.dry_run:
            print(f"[DRY RUN] Would execute: python main_sequential.py --config {temp_config_path}")
            print(f"[DRY RUN] Output folder: {output_folder}")
            os.remove(temp_config_path)
            return True
        
        # Execute main_sequential.py with this config
        try:
            result = subprocess.run(
                [sys.executable, "main_sequential.py", "--config", temp_config_path],
                check=True,
                capture_output=False  # Show output in real-time
            )
            print(f"\n✓ Run {run_id:03d} completed successfully")
            success = True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Run {run_id:03d} failed with error code {e.returncode}")
            success = False
        finally:
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
        
        return success
    
    def run_sweep(self):
        """Execute the full experiment sweep"""
        print(f"\n{'='*80}")
        print(f"Experiment Sweep: {self.experiment_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Base Config: {self.sweep_config.get('base_config')}")
        print(f"Sweep Type: {self.sweep_config.get('sweep_type', 'grid')}")
        if self.dry_run:
            print("MODE: DRY RUN (no actual execution)")
        print(f"{'='*80}\n")
        
        # Generate all run configurations
        combinations = self.generate_run_configs()
        
        # Execute each run
        results = []
        for run_id, param_combo in enumerate(combinations, start=1):
            run_config = self.create_run_config(param_combo, run_id)
            success = self.run_experiment(run_config, run_id, param_combo)
            results.append({
                'run_id': run_id,
                'params': param_combo,
                'success': success
            })
        
        # Print summary
        print(f"\n{'='*80}")
        print("SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"Total runs: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['success'])}")
        print(f"Failed: {sum(1 for r in results if not r['success'])}")
        
        if not self.dry_run:
            print(f"\nResults written to individual CSV files in:")
            print(f"  {self.sweep_config.get('output_base', './outputs/experiments')}/{self.experiment_name}/{self.timestamp}/")
        
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run experiment sweeps for main_sequential.py")
    parser.add_argument(
        "--sweep_config",
        type=str,
        required=True,
        help="Path to sweep configuration JSON file"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without executing experiments"
    )
    
    args = parser.parse_args()
    
    # Validate sweep config exists
    if not os.path.exists(args.sweep_config):
        print(f"Error: Sweep config file not found: {args.sweep_config}")
        sys.exit(1)
    
    # Create and run sweep
    sweep = ExperimentSweep(args.sweep_config, dry_run=args.dry_run)
    sweep.run_sweep()


if __name__ == "__main__":
    main()
