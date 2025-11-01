# Experiment Sweep System

This directory contains sweep configurations for running systematic experiments with `main_sequential.py`.

## Quick Start

```bash
# Run a hyperparameter tuning experiment
python experiment_sweep.py --sweep_config sweep_configs/example_hyperparam_tuning.json

# Run with list mode for specific configurations
python experiment_sweep.py --sweep_config sweep_configs/example_list_mode.json

# Dry run (preview without executing)
python experiment_sweep.py --sweep_config sweep_configs/example_hyperparam_tuning.json --dry_run
```

## Sweep Types

### 1. Grid Search (`"sweep_type": "grid"`)
Tests all combinations of specified parameters. Best for systematic exploration.

**Use for:** Hyperparameter tuning with manageable parameter spaces

**Example:** `example_hyperparam_tuning.json`

```json
{
    "sweep_type": "grid",
    "parameters": {
        "optimization.learning_rate.default": [0.05, 0.1, 0.2],
        "sequential.optimization.decay_rate": [0.95, 0.98]
    }
}
```
This generates 3 × 2 = 6 runs

### 2. List Mode (`"sweep_type": "list"`)
Run specific, hand-picked configurations. Best for targeted comparisons.

**Use for:** Testing specific hypotheses or comparing cherry-picked configurations

**Example:** `example_list_mode.json`

```json
{
    "sweep_type": "list",
    "combinations": [
        {
            "sequential.adaptive_control.enabled": true,
            "optimization.learning_rate.default": 0.1
        },
        {
            "sequential.adaptive_control.enabled": false,
            "optimization.learning_rate.default": 0.2
        }
    ]
}
```

## Configuration Format

### Basic Structure

```json
{
    "experiment_name": "my_experiment",
    "base_config": "configs/sequential.json",
    "sweep_type": "grid",
    "parameters": {
        "path.to.parameter": [value1, value2, value3]
    },
    "fixed_overrides": {
        "path.to.fixed.param": value
    },
    "output_base": "./outputs/experiments"
}
```

### Key Fields

- **`experiment_name`**: Name for this experiment (used in output folder)
- **`base_config`**: Path to base configuration file to modify
- **`sweep_type`**: Type of sweep (`"grid"`, `"list"`, `"random"`, `"ablation"`)
- **`parameters`**: Parameters to sweep over (format depends on sweep_type)
- **`fixed_overrides`**: Parameters to set to fixed values across all runs
- **`output_base`**: Base directory for experiment outputs

### Parameter Path Notation

Use dot notation to specify nested config parameters:

```json
"parameters": {
    "optimization.learning_rate.default": [0.1, 0.2],
    "sequential.adaptive_control.enabled": [true, false],
    "sequential.adaptive_control.opacity_threshold": [0.5, 0.7, 0.9]
}
```

Maps to config structure:
```json
{
    "optimization": {
        "learning_rate": {
            "default": 0.1
        }
    },
    "sequential": {
        "adaptive_control": {
            "enabled": true,
            "opacity_threshold": 0.5
        }
    }
}
```

## Output Organization

Each run creates its own output folder:

```
outputs/experiments/
└── {experiment_name}/
    └── {timestamp}/
        ├── run_001/
        │   ├── run_config.json          # Config used for this run
        │   ├── metrics.csv              # Results (from main_sequential.py)
        │   └── [other outputs...]
        ├── run_002/
        └── run_003/
```

## Common Use Cases

### Hyperparameter Tuning

Find the best learning rate and decay rate:

```json
{
    "experiment_name": "lr_tuning",
    "base_config": "configs/sequential.json",
    "sweep_type": "grid",
    "parameters": {
        "optimization.learning_rate.default": [0.05, 0.1, 0.15, 0.2],
        "sequential.optimization.decay_rate": [0.95, 0.98, 0.99]
    },
    "fixed_overrides": {
        "optimization.num_iterations": 50
    }
}
```

### Adaptive Control Parameter Tuning

Fine-tune adaptive control settings:

```json
{
    "experiment_name": "ac_tuning",
    "base_config": "configs/sequential.json",
    "sweep_type": "grid",
    "parameters": {
        "sequential.adaptive_control.opacity_threshold": [0.5, 0.6, 0.7, 0.8],
        "sequential.adaptive_control.max_primitives_per_tile": [2, 3, 4, 5],
        "sequential.adaptive_control.opacity_reduction_factor": [0.3, 0.5, 0.7]
    },
    "fixed_overrides": {
        "sequential.adaptive_control.enabled": true
    }
}
```

### Quick Comparison

Compare a few specific configurations:

```json
{
    "experiment_name": "quick_comparison",
    "base_config": "configs/sequential.json",
    "sweep_type": "list",
    "combinations": [
        {"optimization.learning_rate.default": 0.1},
        {"optimization.learning_rate.default": 0.2}
    ]
}
```

## Tips

1. **Start with dry run**: Always test with `--dry_run` first to verify configurations
2. **Use shorter iterations for tuning**: Set lower `num_iterations` in `fixed_overrides` for faster experiments
3. **Check output paths**: Each run saves its config in `run_config.json` for reproducibility
4. **Monitor CSV files**: Results are written to `metrics.csv` in each run folder
5. **Random sampling for large spaces**: Use random sampling when grid search would create too many combinations

## Analyzing Results

After running a sweep, you can:

1. **Compare CSV files**: Each run's `metrics.csv` contains all metrics
2. **Use pandas for analysis**:
   ```python
   import pandas as pd
   import glob
   
   # Load all metrics
   csv_files = glob.glob("outputs/experiments/my_experiment/*/run_*/metrics.csv")
   dfs = [pd.read_csv(f) for f in csv_files]
   combined = pd.concat(dfs, ignore_index=True)
   
   # Find best configuration
   best_run = combined.loc[combined['psnr_mean'].idxmax()]
   print(best_run)
   ```

3. **Visualize comparisons**: Plot metrics across different configurations
