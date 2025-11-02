"""
Ablation Study Results Analyzer

Simple analyzer that:
1. Concatenates all CSV files
2. Calculates global PSNR for each experiment
3. Shows summary statistics (mean ± std)
4. Lists all detailed experiment data
5. Exports to clean .txt file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import argparse
from datetime import datetime


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
        # Parse frame PSNR values
        frame_psnrs = [float(x.strip()) for x in frame_psnr_str.split(',')]
        
        # Convert PSNR to MSE for each frame: MSE = 10^(-PSNR/10)
        mses = [10 ** (-psnr / 10) for psnr in frame_psnrs]
        
        # Average MSEs
        mean_mse = np.mean(mses)
        
        # Convert back to PSNR: gPSNR = -10 * log10(mean_MSE)
        global_psnr = -10 * np.log10(mean_mse)
        
        return global_psnr
    except Exception as e:
        print(f"Warning: Could not calculate global PSNR: {e}")
        return np.nan


def load_and_process_csvs(csv_files: List[str]) -> pd.DataFrame:
    """
    Load all CSV files and concatenate them.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Concatenated dataframe with all experiments
    """
    dfs = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Add source file info
            dataset_name = Path(csv_file).stem.replace('metrics_', '')
            df['dataset'] = dataset_name
            
            dfs.append(df)
            print(f"Loaded {len(df)} experiments from {dataset_name}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files loaded")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal experiments loaded: {len(combined_df)}")
    
    # Calculate global PSNR for each row
    if 'frame_psnr' in combined_df.columns:
        print("Calculating global PSNR for each experiment...")
        combined_df['global_psnr'] = combined_df['frame_psnr'].apply(calculate_global_psnr)
        print("Done!")
    else:
        print("Warning: 'frame_psnr' column not found, skipping global PSNR calculation")
    
    return combined_df


def write_group_statistics(f, df_group, metric_cols, group_name):
    """
    Write statistics for a specific group.
    
    Args:
        f: File object
        df_group: DataFrame subset for this group
        metric_cols: List of metric column names
        group_name: Name of the group
    """
    f.write(f"\n{group_name} (n={len(df_group)}):\n")
    f.write("-" * 130 + "\n")
    f.write(f"{'Metric':<15} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}\n")
    f.write("-" * 130 + "\n")
    
    for col in metric_cols:
        if col in df_group.columns:
            mean_val = df_group[col].mean()
            std_val = df_group[col].std()
            min_val = df_group[col].min()
            max_val = df_group[col].max()
            
            # Format metric name
            metric_name = col.replace('avg_', '').replace('_', ' ').upper()
            if col == 'avg_psnr':
                metric_name = 'mPSNR'
            elif col == 'global_psnr':
                metric_name = 'gPSNR'
            
            f.write(f"{metric_name:<15} | ")
            f.write(f"{mean_val:>12.4f} | {std_val:>12.4f} | ")
            f.write(f"{min_val:>12.4f} | {max_val:>12.4f}\n")


def write_summary_report(df: pd.DataFrame, output_file: str):
    """
    Write comprehensive summary report to text file.
    
    Args:
        df: DataFrame with all experiment data
        output_file: Path to output text file
    """
    # Metrics to analyze
    metric_cols = ['avg_psnr', 'avg_ssim', 'avg_vif', 'avg_lpips', 'E_warp', 'tOF', 'tLP']
    if 'global_psnr' in df.columns:
        metric_cols.insert(1, 'global_psnr')
    
    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 130 + "\n")
        f.write("ABLATION STUDY ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}\n")
        f.write("=" * 130 + "\n\n")
        
        # ========================================
        # SUMMARY STATISTICS
        # ========================================
        f.write("=" * 130 + "\n")
        f.write("SUMMARY STATISTICS (All Experiments)\n")
        f.write("=" * 130 + "\n\n")
        
        # Calculate statistics
        f.write(f"{'Metric':<15} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}\n")
        f.write("-" * 130 + "\n")
        
        for col in metric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Format metric name
                metric_name = col.replace('avg_', '').replace('_', ' ').upper()
                if col == 'avg_psnr':
                    metric_name = 'mPSNR'
                elif col == 'global_psnr':
                    metric_name = 'gPSNR'
                
                f.write(f"{metric_name:<15} | ")
                f.write(f"{mean_val:>12.4f} | {std_val:>12.4f} | ")
                f.write(f"{min_val:>12.4f} | {max_val:>12.4f}\n")
        
        f.write("\n\n")
        
        # ========================================
        # ABLATION STATISTICS
        # ========================================
        f.write("=" * 130 + "\n")
        f.write("ABLATION STATISTICS (Grouped by AC and SPO Configuration)\n")
        f.write("=" * 130 + "\n")
        
        # Create 4 groups based on AC and SPO settings
        baseline_group = df[(df['ac_enabled'] == False) & (df['spo_enabled'] == False)]
        ac_only_group = df[(df['ac_enabled'] == True) & (df['spo_enabled'] == False)]
        spo_only_group = df[(df['ac_enabled'] == False) & (df['spo_enabled'] == True)]
        ac_spo_group = df[(df['ac_enabled'] == True) & (df['spo_enabled'] == True)]
        
        # Write statistics for each group
        write_group_statistics(f, baseline_group, metric_cols, "Baseline (AC=No, SPO=No)")
        write_group_statistics(f, ac_only_group, metric_cols, "AC Only (AC=Yes, SPO=No)")
        write_group_statistics(f, spo_only_group, metric_cols, "SPO Only (AC=No, SPO=Yes)")
        write_group_statistics(f, ac_spo_group, metric_cols, "AC+SPO (AC=Yes, SPO=Yes)")
        
        # ========================================
        # IMPROVEMENTS OVER BASELINE
        # ========================================
        if len(baseline_group) > 0:
            f.write("\n\n")
            f.write("=" * 130 + "\n")
            f.write("IMPROVEMENTS OVER BASELINE (%)\n")
            f.write("=" * 130 + "\n\n")
            
            # Calculate baseline means
            baseline_means = {}
            for col in metric_cols:
                if col in baseline_group.columns:
                    baseline_means[col] = baseline_group[col].mean()
            
            # Write comparison header
            f.write(f"{'Group':<20} | ")
            for col in metric_cols:
                if col in df.columns:
                    metric_name = col.replace('avg_', '').replace('_', ' ').upper()
                    if col == 'avg_psnr':
                        metric_name = 'mPSNR'
                    elif col == 'global_psnr':
                        metric_name = 'gPSNR'
                    f.write(f"{metric_name:>10} | ")
            f.write("\n")
            f.write("-" * 130 + "\n")
            
            # Calculate and write improvements for each non-baseline group
            for group_name, group_df in [("AC Only", ac_only_group), 
                                          ("SPO Only", spo_only_group), 
                                          ("AC+SPO", ac_spo_group)]:
                if len(group_df) > 0:
                    f.write(f"{group_name:<20} | ")
                    
                    for col in metric_cols:
                        if col in group_df.columns and col in baseline_means:
                            group_mean = group_df[col].mean()
                            baseline_mean = baseline_means[col]
                            
                            # Calculate % improvement
                            # For LPIPS, E_warp, tOF, tLP: lower is better
                            if col in ['avg_lpips', 'E_warp', 'tOF', 'tLP']:
                                improvement = ((baseline_mean - group_mean) / baseline_mean) * 100
                            else:
                                # For PSNR, SSIM, VIF: higher is better
                                improvement = ((group_mean - baseline_mean) / baseline_mean) * 100
                            
                            f.write(f"{improvement:>+9.2f}% | ")
                    
                    f.write("\n")
        
        f.write("\n\n")
        
        # ========================================
        # DETAILED EXPERIMENT DATA
        # ========================================
        f.write("=" * 130 + "\n")
        f.write("DETAILED EXPERIMENT DATA\n")
        f.write("=" * 130 + "\n\n")
        
        # Write header
        f.write(f"{'#':<4} | {'Dataset':<20} | {'Sequence':<35} | ")
        f.write(f"{'AC':<4} | {'SPO':<4} | ")
        f.write(f"{'mPSNR':<8} | ")
        if 'global_psnr' in df.columns:
            f.write(f"{'gPSNR':<8} | ")
        f.write(f"{'SSIM':<8} | {'VIF':<8} | {'LPIPS':<8} | ")
        f.write(f"{'E_warp':<9} | {'tOF':<8} | {'tLP':<8}\n")
        f.write("-" * 130 + "\n")
        
        # Write each experiment row
        for idx, row in df.iterrows():
            # Extract sequence name from path
            seq_name = Path(row['input_path']).name if pd.notna(row['input_path']) else 'N/A'
            seq_name = seq_name[:35]  # Truncate if too long
            
            # AC and SPO status
            ac_status = 'Yes' if row.get('ac_enabled', False) else 'No'
            spo_status = 'Yes' if row.get('spo_enabled', False) else 'No'
            
            f.write(f"{idx+1:<4} | {row['dataset']:<20} | {seq_name:<35} | ")
            f.write(f"{ac_status:<4} | {spo_status:<4} | ")
            f.write(f"{row['avg_psnr']:>8.3f} | ")
            if 'global_psnr' in df.columns:
                f.write(f"{row['global_psnr']:>8.3f} | ")
            f.write(f"{row['avg_ssim']:>8.4f} | {row['avg_vif']:>8.4f} | ")
            f.write(f"{row['avg_lpips']:>8.4f} | {row['E_warp']:>9.5f} | ")
            f.write(f"{row['tOF']:>8.3f} | {row['tLP']:>8.3f}\n")
        
        f.write("\n")
        f.write("=" * 130 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 130 + "\n")
    
    print(f"\nReport saved to: {output_file}")


def print_console_summary(df: pd.DataFrame):
    """
    Print summary statistics to console.
    
    Args:
        df: DataFrame with all experiment data
    """
    metric_cols = ['avg_psnr', 'avg_ssim', 'avg_vif', 'avg_lpips', 'E_warp', 'tOF', 'tLP']
    if 'global_psnr' in df.columns:
        metric_cols.insert(1, 'global_psnr')
    
    print("\n" + "=" * 130)
    print("SUMMARY STATISTICS (All Experiments)")
    print("=" * 130)
    print(f"\n{'Metric':<15} | {'Mean':<12} | {'Std':<12} | {'Min':<12} | {'Max':<12}")
    print("-" * 130)
    
    for col in metric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            
            metric_name = col.replace('avg_', '').replace('_', ' ').upper()
            if col == 'avg_psnr':
                metric_name = 'mPSNR'
            elif col == 'global_psnr':
                metric_name = 'gPSNR'
            
            print(f"{metric_name:<15} | {mean_val:>12.4f} | {std_val:>12.4f} | "
                  f"{min_val:>12.4f} | {max_val:>12.4f}")
    
    print("\n" + "=" * 130)
    print(f"Total experiments: {len(df)}")
    print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
    print("=" * 130 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation study results from multiple CSV files"
    )
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='One or more CSV files with ablation study results'
    )
    parser.add_argument(
        '--output-file',
        default='./outputs/ablation_analysis/ablation_report.txt',
        help='Output text file path (default: ./outputs/ablation_analysis/ablation_report.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    csv_files = []
    for path in args.csv_files:
        if Path(path).exists():
            csv_files.append(path)
        else:
            print(f"Warning: File not found: {path}")
    
    if not csv_files:
        print("Error: No valid CSV files found")
        return 1
    
    print(f"\nAnalyzing {len(csv_files)} CSV file(s)...\n")
    
    # Load and process all CSVs
    df = load_and_process_csvs(csv_files)
    
    # Print console summary
    print_console_summary(df)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write detailed report
    write_summary_report(df, args.output_file)
    
    print("\nAnalysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
