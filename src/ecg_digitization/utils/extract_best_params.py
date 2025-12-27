"""Utility to extract best hyperparameters from MLflow tuning runs."""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


def get_best_params_from_experiment(
    experiment_name: str,
    tracking_uri: str = "http://mlflow-server:5050",
    metric: str = "best_val_snr",
    minimize: bool = False,
) -> Dict[str, Any]:
    """Get best hyperparameters from MLflow experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        metric: Metric to optimize
        minimize: If True, get run with lowest metric; else highest
        
    Returns:
        Dictionary with best parameters
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Search for runs with the target metric
    order = "ASC" if minimize else "DESC"
    
    # Try multiple metrics as fallback
    metrics_to_try = [metric, "val_snr", "final_val_snr", "val_loss"]
    if metric not in metrics_to_try:
        metrics_to_try.insert(0, metric)
    
    runs = None
    used_metric = metric
    for try_metric in metrics_to_try:
        try:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.{try_metric} IS NOT NULL",
                order_by=[f"metrics.{try_metric} {order}"],
                max_results=1,
            )
            if runs:
                used_metric = try_metric
                break
        except Exception:
            continue
    
    if not runs:
        raise ValueError(f"No runs found with any of metrics: {metrics_to_try}")

    
    best_run = runs[0]
    
    # Extract parameters with 'best_' prefix (from tuning runs)
    best_params = {}
    for key, value in best_run.data.params.items():
        if key.startswith("best_"):
            param_name = key[5:]  # Remove 'best_' prefix
            best_params[param_name] = _parse_value(value)
    
    # Also get direct hyperparameters if no 'best_' params found
    if not best_params:
        hyperparams = [
            "learning_rate", "batch_size", "hidden_dim", "weight_decay",
            "snr_weight", "mse_weight", "augment_prob", "encoder_name"
        ]
        for param in hyperparams:
            if param in best_run.data.params:
                best_params[param] = _parse_value(best_run.data.params[param])
            elif f"best_{param}" in best_run.data.params:
                best_params[param] = _parse_value(best_run.data.params[f"best_{param}"])
    
    # Add run info
    best_params["_run_id"] = best_run.info.run_id
    best_params["_metric_value"] = best_run.data.metrics.get(metric)
    
    return best_params


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    # Try numeric first
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    
    # List
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    return value


def update_config_file(
    config_path: Path,
    params: Dict[str, Any],
    dry_run: bool = True,
) -> None:
    """Update YAML config file with new parameters.
    
    Args:
        config_path: Path to YAML config file
        params: Parameters to update
        dry_run: If True, only print changes without writing
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Map parameter names to config locations
    param_mapping = {
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("data", "batch_size"),
        "weight_decay": ("training", "weight_decay"),
        "hidden_dim": ("model", "hidden_dim"),
        "encoder_name": ("model", "encoder_name"),
        "snr_weight": ("training", "snr_weight"),
        "mse_weight": ("training", "mse_weight"),
        "augment_prob": ("data", "augment_prob"),
    }
    
    changes = []
    for param, value in params.items():
        if param.startswith("_"):
            continue  # Skip metadata
            
        if param in param_mapping:
            section, key = param_mapping[param]
            if section not in config:
                config[section] = {}
            old_value = config[section].get(key)
            config[section][key] = value
            changes.append(f"  {section}.{key}: {old_value} -> {value}")
    
    if changes:
        print("\nProposed changes:")
        for change in changes:
            print(change)
        
        if not dry_run:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"\n✅ Updated {config_path}")
        else:
            print(f"\n[DRY RUN] Would update {config_path}")
            print("Run with --apply to make changes")
    else:
        print("No applicable changes found")


def main():
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters from MLflow and optionally update configs"
    )
    parser.add_argument(
        "--experiment", "-e",
        default="ecg-digitization",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--tracking-uri", "-u",
        default="http://mlflow-server:5050",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--metric", "-m",
        default="best_val_snr",
        help="Metric to optimize"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize metric instead of maximize"
    )
    parser.add_argument(
        "--update-config", "-c",
        type=Path,
        help="Path to config file to update"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (otherwise dry run)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Get best parameters
    best_params = get_best_params_from_experiment(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        metric=args.metric,
        minimize=args.minimize,
    )
    
    if args.json:
        print(json.dumps(best_params, indent=2))
    else:
        print(f"\n🏆 Best Parameters from '{args.experiment}'")
        print(f"   Run ID: {best_params.get('_run_id')}")
        print(f"   {args.metric}: {best_params.get('_metric_value')}")
        print("\nHyperparameters:")
        for key, value in best_params.items():
            if not key.startswith("_"):
                print(f"  {key}: {value}")
    
    # Update config if requested
    if args.update_config:
        update_config_file(
            config_path=args.update_config,
            params=best_params,
            dry_run=not args.apply,
        )


if __name__ == "__main__":
    main()
