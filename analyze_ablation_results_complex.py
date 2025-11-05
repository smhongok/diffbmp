"""
Ablation Study Results Analyzer

Analyzes results from ablation study CSV files and exports comprehensive text summaries.
Compares performance across different configurations of:
- Adaptive Control (AC): enabled/disabled
- Selective Parameter Optimization (SPO): enabled/disabled
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, TextIO
import argparse
from datetime import datetime


class AblationAnalyzer:
    """Analyzer for ablation study results."""
    
    def __init__(self, csv_path: str):
        """
        Initialize analyzer with a CSV file.
        
        Args:
            csv_path: Path to the metrics CSV file
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.dataset_name = Path(csv_path).stem.replace('metrics_', '')
        
        # Clean column names (remove extra spaces)
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Loaded {len(self.df)} experiments from {self.dataset_name}")
    
    def get_configuration_label(self, ac_enabled: bool, spo_enabled: bool) -> str:
        """Get readable label for configuration."""
        if ac_enabled and spo_enabled:
            return "AC+SPO"
        elif ac_enabled and not spo_enabled:
            return "AC only"
        elif not ac_enabled and spo_enabled:
            return "SPO only"
        else:
            return "Baseline"
    
    def calculate_global_psnr(self, frame_psnr_str: str) -> float:
        """
        Calculate global PSNR from per-frame PSNR values.
        
        Global PSNR is calculated by:
        1. Convert each frame PSNR to MSE
        2. Average all MSEs
        3. Convert back to PSNR
        
        Args:
            frame_psnr_str: Comma-separated string of per-frame PSNR values
            
        Returns:
            Global PSNR value
        """
        try:
            # Parse frame PSNR values
            frame_psnrs = [float(x.strip()) for x in frame_psnr_str.split(',')]
            
            # Convert PSNR to MSE for each frame
            # MSE = 10^(-PSNR/10)
            mses = [10 ** (-psnr / 10) for psnr in frame_psnrs]
            
            # Average MSEs
            mean_mse = np.mean(mses)
            
            # Convert back to PSNR
            # PSNR = -10 * log10(MSE)
            global_psnr = -10 * np.log10(mean_mse)
            
            return global_psnr
        except Exception as e:
            print(f"Warning: Could not calculate global PSNR: {e}")
            return np.nan
    
    def summarize_by_configuration(self) -> pd.DataFrame:
        """
        Summarize metrics by AC and SPO configuration.
        
        Returns:
            DataFrame with mean and std metrics for each configuration
        """
        # Calculate global PSNR for each experiment
        if 'frame_psnr' in self.df.columns:
            self.df['global_psnr'] = self.df['frame_psnr'].apply(self.calculate_global_psnr)
        
        grouped = self.df.groupby(['ac_enabled', 'spo_enabled'])
        
        # Determine which metrics to include
        metrics = ['avg_psnr', 'avg_ssim', 'avg_vif', 'avg_lpips', 'E_warp', 'tOF', 'tLP']
        if 'global_psnr' in self.df.columns:
            metrics.insert(1, 'global_psnr')  # Add global_psnr after avg_psnr
        
        # Calculate mean metrics
        summary_mean = grouped[metrics].mean()
        
        # Calculate std metrics
        summary_std = grouped[metrics].std()
        
        # Combine with suffix
        summary = summary_mean.copy()
        for col in summary_mean.columns:
            summary[f'{col}_std'] = summary_std[col]
        
        summary['num_experiments'] = grouped.size()
        
        return summary.reset_index()
    
    def summarize_by_sequence(self) -> pd.DataFrame:
        """
        Summarize metrics by input sequence and configuration.
        
        Returns:
            DataFrame with metrics for each sequence and configuration
        """
        # Calculate global PSNR for each experiment if not already done
        if 'frame_psnr' in self.df.columns and 'global_psnr' not in self.df.columns:
            self.df['global_psnr'] = self.df['frame_psnr'].apply(self.calculate_global_psnr)
        
        grouped = self.df.groupby(['input_path', 'ac_enabled', 'spo_enabled'])
        
        # Determine which metrics to include
        metrics = ['avg_psnr', 'avg_ssim', 'avg_vif', 'avg_lpips', 'E_warp', 'tOF', 'tLP']
        if 'global_psnr' in self.df.columns:
            metrics.insert(1, 'global_psnr')
        
        summary = grouped[metrics].mean()
        
        return summary.reset_index()
    
    def write_summary_to_file(self, file: TextIO):
        """Write comprehensive summary to file."""
        
        # Header
        file.write("=" * 120 + "\n")
        file.write(f"ABLATION STUDY SUMMARY: {self.dataset_name}\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Total experiments: {len(self.df)}\n")
        file.write("=" * 120 + "\n\n")
        
        # Overall configuration summary
        file.write("=" * 120 + "\n")
        file.write("OVERALL SUMMARY BY CONFIGURATION\n")
        file.write("=" * 120 + "\n\n")
        
        summary = self.summarize_by_configuration()
        
        # Write header
        has_global_psnr = 'global_psnr' in summary.columns
        file.write(f"{'Configuration':<15} | ")
        file.write(f"{'mPSNR':<12} | ")
        if has_global_psnr:
            file.write(f"{'gPSNR':<12} | ")
        file.write(f"{'SSIM':<12} | {'VIF':<12} | ")
        file.write(f"{'LPIPS':<12} | {'E_warp':<12} | {'tOF':<12} | {'tLP':<12} | {'n':<4}\n")
        file.write("-" * (120 + (13 if has_global_psnr else 0)) + "\n")
        
        # Write data rows
        for _, row in summary.iterrows():
            config_label = self.get_configuration_label(row['ac_enabled'], row['spo_enabled'])
            file.write(f"{config_label:<15} | ")
            file.write(f"{row['avg_psnr']:>6.3f}±{row['avg_psnr_std']:>4.3f} | ")
            if has_global_psnr:
                file.write(f"{row['global_psnr']:>6.3f}±{row['global_psnr_std']:>4.3f} | ")
            file.write(f"{row['avg_ssim']:>6.4f}±{row['avg_ssim_std']:>4.4f} | ")
            file.write(f"{row['avg_vif']:>6.4f}±{row['avg_vif_std']:>4.4f} | ")
            file.write(f"{row['avg_lpips']:>6.4f}±{row['avg_lpips_std']:>4.4f} | ")
            file.write(f"{row['E_warp']:>7.5f}±{row['E_warp_std']:>4.5f} | ")
            file.write(f"{row['tOF']:>6.3f}±{row['tOF_std']:>4.3f} | ")
            file.write(f"{row['tLP']:>6.3f}±{row['tLP_std']:>4.3f} | ")
            file.write(f"{int(row['num_experiments']):>4}\n")
        
        file.write("\n")
        
        # Calculate improvements
        baseline = summary[(summary['ac_enabled'] == False) & 
                          (summary['spo_enabled'] == False)]
        
        if len(baseline) > 0:
            has_global_psnr = 'global_psnr' in summary.columns
            file.write("=" * (120 + (13 if has_global_psnr else 0)) + "\n")
            file.write("IMPROVEMENTS OVER BASELINE (%)\n")
            file.write("=" * (120 + (13 if has_global_psnr else 0)) + "\n\n")
            
            baseline_metrics = {
                'psnr': baseline['avg_psnr'].values[0],
                'ssim': baseline['avg_ssim'].values[0],
                'vif': baseline['avg_vif'].values[0],
                'lpips': baseline['avg_lpips'].values[0],
                'ewarp': baseline['E_warp'].values[0],
                'tof': baseline['tOF'].values[0],
                'tlp': baseline['tLP'].values[0]
            }
            if has_global_psnr:
                baseline_metrics['gpsnr'] = baseline['global_psnr'].values[0]
            
            file.write(f"{'Configuration':<15} | ")
            file.write(f"{'mPSNR':>8} | ")
            if has_global_psnr:
                file.write(f"{'gPSNR':>8} | ")
            file.write(f"{'SSIM':>8} | {'VIF':>8} | ")
            file.write(f"{'LPIPS':>8} | {'E_warp':>8} | {'tOF':>8} | {'tLP':>8}\n")
            file.write("-" * (120 + (13 if has_global_psnr else 0)) + "\n")
            
            for _, row in summary.iterrows():
                if row['ac_enabled'] or row['spo_enabled']:
                    config_label = self.get_configuration_label(row['ac_enabled'], 
                                                                row['spo_enabled'])
                    
                    # Calculate % improvements (lower is better for lpips, ewarp, tof, tlp)
                    psnr_imp = ((row['avg_psnr'] - baseline_metrics['psnr']) / baseline_metrics['psnr']) * 100
                    ssim_imp = ((row['avg_ssim'] - baseline_metrics['ssim']) / baseline_metrics['ssim']) * 100
                    vif_imp = ((row['avg_vif'] - baseline_metrics['vif']) / baseline_metrics['vif']) * 100
                    lpips_imp = ((baseline_metrics['lpips'] - row['avg_lpips']) / baseline_metrics['lpips']) * 100
                    ewarp_imp = ((baseline_metrics['ewarp'] - row['E_warp']) / baseline_metrics['ewarp']) * 100
                    tof_imp = ((baseline_metrics['tof'] - row['tOF']) / baseline_metrics['tof']) * 100
                    tlp_imp = ((baseline_metrics['tlp'] - row['tLP']) / baseline_metrics['tlp']) * 100
                    
                    file.write(f"  {config_label:12} | ")
                    file.write(f"{psnr_imp:>+7.2f}% | ")
                    if has_global_psnr:
                        gpsnr_imp = ((row['global_psnr'] - baseline_metrics['gpsnr']) / baseline_metrics['gpsnr']) * 100
                        file.write(f"{gpsnr_imp:>+7.2f}% | ")
                    file.write(f"{ssim_imp:>+7.2f}% | "
                          f"{vif_imp:>+7.2f}% | "
                          f"{lpips_imp:>+7.2f}% | "
                          f"{ewarp_imp:>+7.2f}% | "
                          f"{tof_imp:>+7.2f}% | "
                          f"{tlp_imp:>+7.2f}%\n")
            
            file.write("\n")
        
        # Per-sequence breakdown
        seq_summary = self.summarize_by_sequence()
        has_global_psnr = 'global_psnr' in seq_summary.columns
        
        file.write("=" * (120 + (11 if has_global_psnr else 0)) + "\n")
        file.write("PER-SEQUENCE BREAKDOWN\n")
        file.write("=" * (120 + (11 if has_global_psnr else 0)) + "\n\n")
        
        # Group by sequence
        for sequence in seq_summary['input_path'].unique():
            seq_data = seq_summary[seq_summary['input_path'] == sequence]
            
            file.write(f"\nSequence: {sequence}\n")
            file.write("-" * (120 + (11 if has_global_psnr else 0)) + "\n")
            file.write(f"{'Configuration':<15} | ")
            file.write(f"{'mPSNR':>8} | ")
            if has_global_psnr:
                file.write(f"{'gPSNR':>8} | ")
            file.write(f"{'SSIM':>8} | {'VIF':>8} | ")
            file.write(f"{'LPIPS':>8} | {'E_warp':>8} | {'tOF':>8} | {'tLP':>8}\n")
            file.write("-" * (120 + (11 if has_global_psnr else 0)) + "\n")
            
            for _, row in seq_data.iterrows():
                config_label = self.get_configuration_label(row['ac_enabled'], row['spo_enabled'])
                file.write(f"{config_label:<15} | ")
                file.write(f"{row['avg_psnr']:>8.3f} | ")
                if has_global_psnr:
                    file.write(f"{row['global_psnr']:>8.3f} | ")
                file.write(f"{row['avg_ssim']:>8.4f} | {row['avg_vif']:>8.4f} | ")
                file.write(f"{row['avg_lpips']:>8.4f} | {row['E_warp']:>8.5f} | {row['tOF']:>8.3f} | {row['tLP']:>8.3f}\n")
            
            file.write("\n")
        
        # Raw data table
        has_global_psnr = 'global_psnr' in self.df.columns
        file.write("=" * (120 + (11 if has_global_psnr else 0)) + "\n")
        file.write("RAW EXPERIMENT DATA\n")
        file.write("=" * (120 + (11 if has_global_psnr else 0)) + "\n\n")
        
        file.write(f"{'Sequence':<40} | {'Config':<12} | ")
        file.write(f"{'mPSNR':>8} | ")
        if has_global_psnr:
            file.write(f"{'gPSNR':>8} | ")
        file.write(f"{'SSIM':>8} | {'VIF':>8} | ")
        file.write(f"{'LPIPS':>8} | {'E_warp':>8} | {'tOF':>8} | {'tLP':>8}\n")
        file.write("-" * (120 + (11 if has_global_psnr else 0)) + "\n")
        
        for _, row in self.df.iterrows():
            config_label = self.get_configuration_label(row['ac_enabled'], row['spo_enabled'])
            seq_name = Path(row['input_path']).name if pd.notna(row['input_path']) else 'N/A'
            file.write(f"{seq_name:<40} | {config_label:<12} | ")
            file.write(f"{row['avg_psnr']:>8.3f} | ")
            if has_global_psnr:
                file.write(f"{row['global_psnr']:>8.3f} | ")
            file.write(f"{row['avg_ssim']:>8.4f} | {row['avg_vif']:>8.4f} | ")
            file.write(f"{row['avg_lpips']:>8.4f} | {row['E_warp']:>8.5f} | {row['tOF']:>8.3f} | {row['tLP']:>8.3f}\n")
        
        file.write("\n")
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.summarize_by_configuration()
        has_global_psnr = 'global_psnr' in summary.columns
        
        width = 120 + (13 if has_global_psnr else 0)
        print(f"\n{'='*width}")
        print(f"Dataset: {self.dataset_name}")
        print(f"{'='*width}\n")
        
        # Print header
        print(f"{'Configuration':<15} | ", end="")
        print(f"{'mPSNR':<12} | ", end="")
        if has_global_psnr:
            print(f"{'gPSNR':<12} | ", end="")
        print(f"{'SSIM':<12} | {'VIF':<12} | ", end="")
        print(f"{'LPIPS':<12} | {'E_warp':<12} | {'tOF':<12} | {'tLP':<12} | {'n':<4}")
        print("-" * width)
        
        # Print data
        for _, row in summary.iterrows():
            config_label = self.get_configuration_label(row['ac_enabled'], row['spo_enabled'])
            print(f"{config_label:<15} | ", end="")
            print(f"{row['avg_psnr']:>6.3f}±{row['avg_psnr_std']:>4.3f} | ", end="")
            if has_global_psnr:
                print(f"{row['global_psnr']:>6.3f}±{row['global_psnr_std']:>4.3f} | ", end="")
            print(f"{row['avg_ssim']:>6.4f}±{row['avg_ssim_std']:>4.4f} | ", end="")
            print(f"{row['avg_vif']:>6.4f}±{row['avg_vif_std']:>4.4f} | ", end="")
            print(f"{row['avg_lpips']:>6.4f}±{row['avg_lpips_std']:>4.4f} | ", end="")
            print(f"{row['E_warp']:>7.5f}±{row['E_warp_std']:>4.5f} | ", end="")
            print(f"{row['tOF']:>6.3f}±{row['tOF_std']:>4.3f} | ", end="")
            print(f"{row['tLP']:>6.3f}±{row['tLP_std']:>4.3f} | ", end="")
            print(f"{int(row['num_experiments']):>4}")
        
        print()


class MultiDatasetAnalyzer:
    """Analyzer for multiple datasets."""
    
    def __init__(self, csv_paths: List[str]):
        """
        Initialize with multiple CSV files.
        
        Args:
            csv_paths: List of paths to metrics CSV files
        """
        self.analyzers = [AblationAnalyzer(path) for path in csv_paths]
        self.dataset_names = [a.dataset_name for a in self.analyzers]
    
    def write_combined_summary(self, output_path: str):
        """
        Write combined summary for all datasets to a single file.
        
        Args:
            output_path: Path to output text file
        """
        with open(output_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("MULTI-DATASET ABLATION STUDY SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datasets: {', '.join(self.dataset_names)}\n")
            f.write("=" * 120 + "\n\n")
            
            # Write summary for each dataset
            for analyzer in self.analyzers:
                analyzer.write_summary_to_file(f)
                f.write("\n" + "=" * 120 + "\n\n")
            
            # Cross-dataset comparison
            f.write("=" * 120 + "\n")
            f.write("CROSS-DATASET COMPARISON\n")
            f.write("=" * 120 + "\n\n")
            
            all_summaries = []
            for analyzer in self.analyzers:
                summary = analyzer.summarize_by_configuration()
                summary['dataset'] = analyzer.dataset_name
                all_summaries.append(summary)
            
            combined = pd.concat(all_summaries, ignore_index=True)
            
            # Group by configuration and show average across datasets
            for config in [(False, False), (True, False), (False, True), (True, True)]:
                ac_enabled, spo_enabled = config
                config_data = combined[(combined['ac_enabled'] == ac_enabled) & 
                                      (combined['spo_enabled'] == spo_enabled)]
                
                if len(config_data) > 0:
                    config_label = self.analyzers[0].get_configuration_label(ac_enabled, spo_enabled)
                    
                    f.write(f"\n{config_label}:\n")
                    f.write("-" * 120 + "\n")
                    f.write(f"{'Dataset':<15} | ")
                    f.write(f"{'PSNR':>8} | {'SSIM':>8} | {'VIF':>8} | ")
                    f.write(f"{'LPIPS':>8} | {'tLP':>8}\n")
                    f.write("-" * 120 + "\n")
                    
                    for _, row in config_data.iterrows():
                        f.write(f"{row['dataset']:<15} | ")
                        f.write(f"{row['avg_psnr']:>8.3f} | {row['avg_ssim']:>8.4f} | {row['avg_vif']:>8.4f} | ")
                        f.write(f"{row['avg_lpips']:>8.4f} | {row['tLP']:>8.3f}\n")
                    
                    # Average across datasets
                    f.write("-" * 120 + "\n")
                    f.write(f"{'Average':<15} | ")
                    f.write(f"{config_data['avg_psnr'].mean():>8.3f} | ")
                    f.write(f"{config_data['avg_ssim'].mean():>8.4f} | ")
                    f.write(f"{config_data['avg_vif'].mean():>8.4f} | ")
                    f.write(f"{config_data['avg_lpips'].mean():>8.4f} | ")
                    f.write(f"{config_data['tLP'].mean():>8.3f}\n")
                    f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation study results from CSV files"
    )
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='One or more CSV files with ablation study results'
    )
    parser.add_argument(
        '--output-dir',
        default='./outputs/ablation_analysis',
        help='Directory to save analysis (default: ./outputs/ablation_analysis)'
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(csv_files)} dataset(s)...\n")
    
    if len(csv_files) == 1:
        # Single dataset analysis
        analyzer = AblationAnalyzer(csv_files[0])
        analyzer.print_summary()
        
        # Write detailed report
        output_file = output_dir / f'ablation_report_{analyzer.dataset_name}.txt'
        with open(output_file, 'w') as f:
            analyzer.write_summary_to_file(f)
        
        print(f"\nDetailed report saved to: {output_file}")
        
    else:
        # Multi-dataset analysis
        multi_analyzer = MultiDatasetAnalyzer(csv_files)
        
        # Print summaries for all datasets
        for analyzer in multi_analyzer.analyzers:
            analyzer.print_summary()
        
        # Write individual reports
        for analyzer in multi_analyzer.analyzers:
            output_file = output_dir / f'ablation_report_{analyzer.dataset_name}.txt'
            with open(output_file, 'w') as f:
                analyzer.write_summary_to_file(f)
            print(f"Report saved: {output_file}")
        
        # Write combined report
        combined_output = output_dir / 'ablation_report_combined.txt'
        multi_analyzer.write_combined_summary(str(combined_output))
        print(f"\nCombined report saved to: {combined_output}")
    
    print(f"\nAnalysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
