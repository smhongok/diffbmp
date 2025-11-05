"""
Hyperparameter Tuning Results Analyzer

Analyzes hyperparameter tuning results from ablation study CSV files.
For each hyperparameter that was swept, compares different values against baseline.

Usage:
    python analyze_hyperparameter_tuning.py --csv metrics_tuning_study_1.csv --config sweep_configs/tuning_study_1.json
    python analyze_hyperparameter_tuning.py --csv outputs/metrics_*.csv --config sweep_configs/tuning_study_1.json --baseline-config configs/sequential.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from datetime import datetime
import glob


def calculate_global_psnr(frame_psnr_str: str) -> float:
    """
    Calculate global PSNR from per-frame PSNR values.
    
    Global PSNR = -10 * log10(mean(10^(-PSNR/10)))
    
    Args:
        frame_psnr_str: Comma-separated string of per-frame PSNR values
        
    Returns:
        Global PSNR value
    """
    try:
        frame_psnrs = [float(x.strip()) for x in frame_psnr_str.split(',')]
        mses = [10 ** (-psnr / 10) for psnr in frame_psnrs]
        mean_mse = np.mean(mses)
        global_psnr = -10 * np.log10(mean_mse)
        return global_psnr
    except Exception as e:
        print(f"Warning: Could not calculate global PSNR: {e}")
        return np.nan


def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """Load sweep configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_baseline_config(config_path: str) -> Dict[str, Any]:
    """Load baseline configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_nested_value(config: Dict, key_path: str) -> Any:
    """Get value from nested config using dot notation (e.g., 'sequential.adaptive_control.enabled')."""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def map_config_param_to_csv_column(param_name: str) -> str:
    """
    Map configuration parameter names to CSV column names.
    
    Args:
        param_name: Config parameter name (e.g., 'initialization.N')
        
    Returns:
        CSV column name (e.g., 'num_splats')
    """
    # Mapping from config parameter paths to CSV column names
    mapping = {
        'initialization.N': 'num_splats',
        'initialization.radii_min': 'radii_min',  # May not exist in CSV
        'sequential.adaptive_control.enabled': 'ac_enabled',
        'sequential.selective_parameter_optimization.enabled': 'spo_enabled',
        'sequential.adaptive_control.min_criteria_count': 'ac_min_criteria_count',
        'sequential.adaptive_control.apply_epochs': 'ac_apply_epochs',
    }
    
    return mapping.get(param_name, param_name)


def identify_hyperparameters(sweep_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Identify which parameters are hyperparameters (not dataset paths or feature flags).
    
    Returns:
        Dictionary of hyperparameter_name -> list of values
    """
    parameters = sweep_config.get('parameters', {})
    hyperparameters = {}
    
    # Exclude dataset paths
    dataset_keys = ['preprocessing.img_path', 'img_path']
    
    # Identify feature flags (boolean parameters)
    for key, values in parameters.items():
        if key in dataset_keys:
            continue
        # If all values are boolean, it's a feature flag
        if all(isinstance(v, bool) for v in values):
            continue  # Skip feature flags
        else:
            hyperparameters[key] = values
    
    return hyperparameters


def identify_feature_flags(sweep_config: Dict[str, Any]) -> Dict[str, List[bool]]:
    """
    Identify feature flags (boolean ablation study parameters).
    
    Returns:
        Dictionary of feature_name -> list of boolean values
    """
    parameters = sweep_config.get('parameters', {})
    feature_flags = {}
    
    # Exclude dataset paths
    dataset_keys = ['preprocessing.img_path', 'img_path']
    
    for key, values in parameters.items():
        if key in dataset_keys:
            continue
        # If all values are boolean, it's a feature flag
        if all(isinstance(v, bool) for v in values):
            feature_flags[key] = values
    
    return feature_flags


def load_csv_files(csv_pattern: str) -> pd.DataFrame:
    """
    Load CSV file(s) matching the pattern.
    
    Args:
        csv_pattern: Path or glob pattern to CSV file(s)
        
    Returns:
        Combined dataframe
    """
    csv_files = glob.glob(csv_pattern)
    if not csv_files:
        raise ValueError(f"No CSV files found matching: {csv_pattern}")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            
            # Add source file info
            dataset_name = Path(csv_file).stem.replace('metrics_', '')
            df['source_file'] = dataset_name
            
            dfs.append(df)
            print(f"Loaded {len(df)} experiments from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total experiments loaded: {len(combined_df)}\n")
    
    # Calculate global PSNR
    if 'frame_psnr' in combined_df.columns:
        combined_df['global_psnr'] = combined_df['frame_psnr'].apply(calculate_global_psnr)
    
    return combined_df


def get_feature_combination_name(feature_values: Dict[str, bool]) -> str:
    """Generate a readable name for the feature combination."""
    # Expected features: adaptive_control.enabled and selective_parameter_optimization.enabled
    ac_enabled = feature_values.get('sequential.adaptive_control.enabled', False)
    spo_enabled = feature_values.get('sequential.selective_parameter_optimization.enabled', False)
    
    if not ac_enabled and not spo_enabled:
        return "Baseline (no AC, no SPO)"
    elif ac_enabled and not spo_enabled:
        return "AC only"
    elif not ac_enabled and spo_enabled:
        return "SPO only"
    elif ac_enabled and spo_enabled:
        return "AC + SPO"
    else:
        # Fallback for unexpected combinations
        parts = []
        for feature_name, value in sorted(feature_values.items()):
            simple_name = feature_name.split('.')[-1].replace('.enabled', '')
            if value:
                parts.append(simple_name)
        return " + ".join(parts) if parts else "unknown"


def analyze_hyperparameters(
    df: pd.DataFrame,
    hyperparameters: Dict[str, List[Any]],
    feature_flags: Dict[str, List[bool]]
) -> Dict[str, Any]:
    """
    Analyze the effect of hyperparameter combinations.
    
    For each combination of hyperparameter values, compare AC, SPO, and AC+SPO against baseline (no AC, no SPO).
    
    Args:
        df: DataFrame with experiment results
        hyperparameters: Dictionary of hyperparameter_name -> list of values
        feature_flags: Dictionary of feature flags
        
    Returns:
        Analysis results dictionary with comparisons for each hyperparameter combination
    """
    # Map config parameter names to CSV column names
    param_name_to_csv = {}
    for param_name in hyperparameters.keys():
        csv_column = map_config_param_to_csv_column(param_name)
        if csv_column not in df.columns:
            print(f"  Warning: Column '{csv_column}' not found in CSV. Skipping {param_name}.")
        else:
            param_name_to_csv[param_name] = csv_column
    
    if not param_name_to_csv:
        return {
            'hyperparameters': hyperparameters,
            'combinations': []
        }
    
    results = {
        'hyperparameters': hyperparameters,
        'param_name_to_csv': param_name_to_csv,
        'combinations': []  # One entry per hyperparameter combination
    }
    
    # Map feature flag names to CSV column names
    csv_feature_flags = {map_config_param_to_csv_column(k): v for k, v in feature_flags.items()}
    
    # Define baseline feature values (all features disabled)
    baseline_features = {map_config_param_to_csv_column(key): False for key in feature_flags.keys()}
    
    # Generate all combinations of hyperparameter values
    from itertools import product
    param_names = list(param_name_to_csv.keys())
    param_value_lists = [hyperparameters[pn] for pn in param_names]
    
    for param_value_tuple in product(*param_value_lists):
        # Create combination dictionary
        combination = {param_names[i]: param_value_tuple[i] for i in range(len(param_names))}
        
        combination_comparison = {
            'combination': combination,
            'baseline': None,
            'feature_comparisons': []  # AC only, SPO only, AC+SPO
        }
        
        # Filter data for this hyperparameter combination
        combo_df = df
        for param_name, param_value in combination.items():
            csv_column = param_name_to_csv[param_name]
            # Handle list values - CSV stores them as strings
            if isinstance(param_value, list):
                param_value_str = str(param_value)
                combo_df = combo_df[combo_df[csv_column].astype(str) == param_value_str]
            else:
                combo_df = combo_df[combo_df[csv_column] == param_value]
        
        if len(combo_df) == 0:
            continue
        
        # Get baseline metrics (no AC, no SPO)
        baseline_df = combo_df
        for feature_name, feature_value in baseline_features.items():
            baseline_df = baseline_df[baseline_df[feature_name] == feature_value]
        
        if len(baseline_df) == 0:
            combo_str = ', '.join([f"{k}={v}" for k, v in combination.items()])
            print(f"  Warning: No baseline data for combination: {combo_str}")
            continue
        
        # Define all metrics to analyze
        metric_names = ['global_psnr', 'avg_psnr', 'avg_ssim', 'avg_vif', 'avg_lpips', 'E_warp', 'tOF', 'tLP']
        
        baseline_metrics = {'count': len(baseline_df)}
        for metric in metric_names:
            if metric in baseline_df.columns:
                baseline_metrics[f'{metric}_mean'] = baseline_df[metric].mean()
                baseline_metrics[f'{metric}_std'] = baseline_df[metric].std()
        
        combination_comparison['baseline'] = baseline_metrics
        
        # Compare each feature combination against baseline
        # Generate all possible feature combinations using CSV column names
        csv_feature_names = list(csv_feature_flags.keys())
        config_feature_names = list(feature_flags.keys())
        
        if len(csv_feature_names) == 2:
            # Typical case: AC and SPO
            feature_combinations_to_test = [
                {csv_feature_names[0]: True, csv_feature_names[1]: False},   # AC only
                {csv_feature_names[0]: False, csv_feature_names[1]: True},   # SPO only
                {csv_feature_names[0]: True, csv_feature_names[1]: True},    # AC + SPO
            ]
        else:
            # General case: generate all non-baseline combinations
            from itertools import product
            feature_combinations_to_test = []
            for combo in product([True, False], repeat=len(csv_feature_names)):
                combo_dict = dict(zip(csv_feature_names, combo))
                # Skip baseline (all False)
                if not all(v == False for v in combo_dict.values()):
                    feature_combinations_to_test.append(combo_dict)
        
        for feature_combo in feature_combinations_to_test:
            # Filter data for this feature combination
            feature_combo_df = combo_df
            for feature_name, feature_value in feature_combo.items():
                feature_combo_df = feature_combo_df[feature_combo_df[feature_name] == feature_value]
            
            if len(feature_combo_df) == 0:
                continue
            
            # Convert CSV column names back to config names for display
            config_feature_combo = {}
            for csv_col, val in feature_combo.items():
                for config_name, csv_name in zip(config_feature_names, csv_feature_names):
                    if csv_col == csv_name:
                        config_feature_combo[config_name] = val
                        break
            
            combo_name = get_feature_combination_name(config_feature_combo)
            
            metrics = {
                'feature_combination': combo_name,
                'feature_values': config_feature_combo,
                'count': len(feature_combo_df)
            }
            
            # Calculate metrics and deltas vs baseline
            for metric in metric_names:
                if metric in feature_combo_df.columns:
                    metrics[f'{metric}_mean'] = feature_combo_df[metric].mean()
                    metrics[f'{metric}_std'] = feature_combo_df[metric].std()
                    # Calculate delta
                    if f'{metric}_mean' in baseline_metrics:
                        metrics[f'{metric}_delta'] = metrics[f'{metric}_mean'] - baseline_metrics[f'{metric}_mean']
            
            combination_comparison['feature_comparisons'].append(metrics)
        
        if combination_comparison['baseline'] is not None:
            results['combinations'].append(combination_comparison)
    
    return results


def write_hyperparameter_report(
    analysis_results: Dict[str, Any],
    output_file: str,
    sweep_config: Dict[str, Any]
):
    """
    Write comprehensive hyperparameter tuning report to text file.
    
    Args:
        analysis_results: Analysis results with hyperparameter combinations
        output_file: Path to output text file
        sweep_config: Original sweep configuration
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment: {sweep_config.get('experiment_name', 'Unknown')}\n")
        f.write(f"Base Config: {sweep_config.get('base_config', 'Unknown')}\n")
        f.write("\n")
        f.write("BASELINE: No Adaptive Control (AC), No Selective Parameter Optimization (SPO)\n")
        f.write("\n")
        
        # Summary of hyperparameters
        f.write("HYPERPARAMETERS ANALYZED:\n")
        f.write("-" * 80 + "\n")
        hyperparameters = analysis_results['hyperparameters']
        for param_name, values in hyperparameters.items():
            f.write(f"  • {param_name}: {values}\n")
        f.write("\n\n")
        
        # Detailed analysis for each hyperparameter combination
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER COMBINATIONS\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        # Analysis for each combination
        for combination_comparison in analysis_results['combinations']:
            combination = combination_comparison['combination']
            baseline = combination_comparison['baseline']
                
            # Format combination as a title
            combo_parts = []
            for param_name, param_value in combination.items():
                simple_name = param_name.split('.')[-1]
                combo_parts.append(f"{simple_name}={param_value}")
            combo_title = ", ".join(combo_parts)
            
            f.write("-" * 80 + "\n")
            f.write(f"COMBINATION: {combo_title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Baseline (no AC, no SPO) - {baseline['count']} experiments:\n")
            
            # Write all baseline metrics
            metric_display = [
                ('gPSNR', 'global_psnr'),
                ('mPSNR', 'avg_psnr'),
                ('SSIM', 'avg_ssim'),
                ('VIF', 'avg_vif'),
                ('LPIPS', 'avg_lpips'),
                ('E_warp', 'E_warp'),
                ('tOF', 'tOF'),
                ('tLP', 'tLP')
            ]
            
            for display_name, metric_key in metric_display:
                if f'{metric_key}_mean' in baseline:
                    mean_val = baseline[f'{metric_key}_mean']
                    std_val = baseline[f'{metric_key}_std']
                    f.write(f"  {display_name:10s}: {mean_val:>8.4f} ± {std_val:<8.4f}\n")
                
            f.write("\n")
            
            if not combination_comparison['feature_comparisons']:
                f.write("  No feature combinations to compare.\n\n")
                continue
            
            f.write("Comparisons vs Baseline:\n")
            for metrics in combination_comparison['feature_comparisons']:
                feature_combo = metrics['feature_combination']
                
                f.write(f"\n  {feature_combo} - {metrics['count']} experiments:\n")
                
                # Write all metrics with deltas
                for display_name, metric_key in metric_display:
                    if f'{metric_key}_mean' in metrics:
                        mean_val = metrics[f'{metric_key}_mean']
                        std_val = metrics[f'{metric_key}_std']
                        
                        if f'{metric_key}_delta' in metrics:
                            delta = metrics[f'{metric_key}_delta']
                            delta_sign = "+" if delta >= 0 else ""
                            f.write(f"    {display_name:10s}: {mean_val:>8.4f} ± {std_val:<8.4f}  ({delta_sign}{delta:>8.4f})\n")
                        else:
                            f.write(f"    {display_name:10s}: {mean_val:>8.4f} ± {std_val:<8.4f}\n")
            
            f.write("\n")
        
        # Summary recommendations
        f.write("=" * 80 + "\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        # Find best combination for each feature combination (by gPSNR)
        feature_combo_best = {}  # feature_combo -> (best_combination, best_psnr_delta)
        
        for combination_comparison in analysis_results['combinations']:
            combination = combination_comparison['combination']
            combo_str = ', '.join([f"{k.split('.')[-1]}={v}" for k, v in combination.items()])
            
            for metrics in combination_comparison['feature_comparisons']:
                feature_combo = metrics['feature_combination']
                if 'global_psnr_delta' in metrics:
                    psnr_delta = metrics['global_psnr_delta']
                    
                    if feature_combo not in feature_combo_best:
                        feature_combo_best[feature_combo] = (combo_str, psnr_delta)
                    else:
                        if psnr_delta > feature_combo_best[feature_combo][1]:
                            feature_combo_best[feature_combo] = (combo_str, psnr_delta)
        
        if feature_combo_best:
            for feature_combo, (best_combo, best_delta) in sorted(feature_combo_best.items()):
                f.write(f"• Best gPSNR for {feature_combo}:\n")
                f.write(f"  {best_combo} ({best_delta:+.4f} dB)\n")
                f.write("\n")
        else:
            f.write("• No sufficient data for recommendations\n")


def print_console_summary(analysis_results: Dict[str, Any]):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING ANALYSIS SUMMARY")
    print("=" * 80)
    print("Baseline: No AC, No SPO")
    print("=" * 80)
    
    hyperparameters = analysis_results['hyperparameters']
    print(f"\nHyperparameters: {', '.join([k.split('.')[-1] for k in hyperparameters.keys()])}")
    print("-" * 80)
    
    for combination_comparison in analysis_results['combinations']:
        combination = combination_comparison['combination']
        baseline = combination_comparison['baseline']
        
        # Format combination
        combo_parts = [f"{k.split('.')[-1]}={v}" for k, v in combination.items()]
        combo_str = ", ".join(combo_parts)
        
        print(f"\n  {combo_str}")
        
        # Print baseline metrics
        baseline_str = "Baseline: "
        if 'global_psnr_mean' in baseline:
            baseline_str += f"gPSNR {baseline['global_psnr_mean']:.4f}"
        if 'avg_ssim_mean' in baseline:
            baseline_str += f", SSIM {baseline['avg_ssim_mean']:.4f}"
        baseline_str += f" ({baseline['count']} exp)"
        print(f"    {baseline_str}")
        
        if not combination_comparison['feature_comparisons']:
            print("    No comparisons available")
            continue
        
        for metrics in combination_comparison['feature_comparisons']:
            feature_combo = metrics['feature_combination']
            
            # Build comparison string with key metrics
            comp_parts = []
            if 'global_psnr_delta' in metrics:
                delta = metrics['global_psnr_delta']
                sign = "+" if delta >= 0 else ""
                comp_parts.append(f"gPSNR {sign}{delta:.4f}")
            if 'avg_ssim_delta' in metrics:
                delta = metrics['avg_ssim_delta']
                sign = "+" if delta >= 0 else ""
                comp_parts.append(f"SSIM {sign}{delta:.4f}")
            if 'avg_lpips_delta' in metrics:
                delta = metrics['avg_lpips_delta']
                sign = "+" if delta >= 0 else ""
                comp_parts.append(f"LPIPS {sign}{delta:.4f}")
            
            comp_str = ", ".join(comp_parts) if comp_parts else "No metrics"
            print(f"    {feature_combo:20s}: {comp_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter tuning results from ablation study CSV files"
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='Path or glob pattern to CSV file(s) (e.g., outputs/metrics_tuning_study_1.csv or outputs/metrics_*.csv)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to sweep configuration JSON file'
    )
    parser.add_argument(
        '--baseline-config',
        help='Path to baseline configuration file (optional, for baseline value reference)'
    )
    parser.add_argument(
        '--output',
        help='Output text file path (default: hyperparameter_tuning_analysis_<timestamp>.txt)'
    )
    
    args = parser.parse_args()
    
    # Load sweep config
    print("Loading sweep configuration...")
    sweep_config = load_sweep_config(args.config)
    print(f"Experiment: {sweep_config.get('experiment_name', 'Unknown')}\n")
    
    # Load baseline config if provided
    baseline_config = None
    if args.baseline_config:
        print("Loading baseline configuration...")
        baseline_config = load_baseline_config(args.baseline_config)
    
    # Identify hyperparameters and feature flags
    hyperparameters = identify_hyperparameters(sweep_config)
    feature_flags = identify_feature_flags(sweep_config)
    
    print(f"Identified {len(hyperparameters)} hyperparameter(s):")
    for param_name, values in hyperparameters.items():
        print(f"  • {param_name}: {values}")
    print()
    
    print(f"Identified {len(feature_flags)} feature flag(s):")
    for feature_name, values in feature_flags.items():
        print(f"  • {feature_name}: {values}")
    print()
    
    # Load CSV data
    print("Loading experiment results...")
    df = load_csv_files(args.csv)
    
    # Analyze hyperparameter combinations
    print(f"Analyzing hyperparameter combinations...")
    analysis_results = analyze_hyperparameters(
        df, hyperparameters, feature_flags
    )
    
    # Write report
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'hyperparameter_tuning_analysis_{timestamp}.txt'
    
    print(f"\nWriting report to {output_file}...")
    write_hyperparameter_report(analysis_results, output_file, sweep_config)
    
    # Print console summary
    print_console_summary(analysis_results)
    
    print(f"\n✓ Analysis complete! Report saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
