"""
Automated experiment report generator.

Pulls data from MLflow and generates comprehensive markdown reports.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
from mlflow.entities import Run
import pandas as pd
from loguru import logger


class ExperimentReportGenerator:
    """Generate comprehensive reports from MLflow experiments."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5050",
        reports_dir: str = "reports",
    ):
        """Initialize report generator.
        
        Args:
            tracking_uri: MLflow tracking URI
            reports_dir: Directory to save reports
        """
        self.tracking_uri = tracking_uri
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(tracking_uri)
    
    def generate_run_report(
        self,
        run_id: str,
        experiment_name: Optional[str] = None,
    ) -> str:
        """Generate report for a specific run.
        
        Args:
            run_id: MLflow run ID
            experiment_name: Optional experiment name
            
        Returns:
            Path to generated report
        """
        # Get run data
        run = mlflow.get_run(run_id)
        
        # Generate report
        report = self._create_report_markdown(run, experiment_name)
        
        # Save report
        report_filename = f"run_{run_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Generated report: {report_path}")
        return str(report_path)
    
    def generate_experiment_summary(
        self,
        experiment_name: str,
        top_n: int = 10,
    ) -> str:
        """Generate summary report for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            top_n: Number of top runs to include
            
        Returns:
            Path to generated report
        """
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_snr DESC"],
            max_results=top_n,
        )
        
        # Generate report
        report = self._create_experiment_summary_markdown(
            experiment_name,
            runs,
            top_n,
        )
        
        # Save report
        report_filename = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Generated experiment summary: {report_path}")
        return str(report_path)
    
    def _create_report_markdown(
        self,
        run: Run,
        experiment_name: Optional[str] = None,
    ) -> str:
        """Create markdown report for a run.
        
        Args:
            run: MLflow run object
            experiment_name: Optional experiment name
            
        Returns:
            Markdown report string
        """
        info = run.info
        data = run.data
        
        # Extract key information
        run_id = info.run_id
        run_name = data.tags.get("mlflow.runName", "Unnamed Run")
        status = info.status
        start_time = datetime.fromtimestamp(info.start_time / 1000)
        end_time = datetime.fromtimestamp(info.end_time / 1000) if info.end_time else None
        duration = (end_time - start_time) if end_time else None
        
        # Build report
        report = f"""# Experiment Run Report

## Run Information

| Property | Value |
|----------|-------|
| **Run ID** | `{run_id}` |
| **Run Name** | {run_name} |
| **Experiment** | {experiment_name or 'N/A'} |
| **Status** | {status} |
| **Start Time** | {start_time.strftime('%Y-%m-%d %H:%M:%S')} |
| **End Time** | {end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'Running'} |
| **Duration** | {self._format_duration(duration) if duration else 'N/A'} |

---

## Tags

"""
        # Add tags
        tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
        if tags:
            for key, value in sorted(tags.items()):
                report += f"- **{key}**: {value}\n"
        else:
            report += "*No custom tags*\n"
        
        report += "\n---\n\n## Configuration Parameters\n\n"
        
        # Add parameters (grouped by category)
        params = data.params
        if params:
            param_groups = self._group_parameters(params)
            for group_name, group_params in param_groups.items():
                report += f"### {group_name}\n\n"
                report += "| Parameter | Value |\n"
                report += "|-----------|-------|\n"
                for key, value in sorted(group_params.items()):
                    report += f"| `{key}` | {value} |\n"
                report += "\n"
        else:
            report += "*No parameters logged*\n"
        
        report += "---\n\n## Performance Metrics\n\n"
        
        # Add metrics
        metrics = data.metrics
        if metrics:
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            
            # Prioritize important metrics
            priority_metrics = ["val_snr", "val_loss", "train_loss", "best_val_loss"]
            for metric in priority_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    report += f"| **{metric}** | {value:.4f} |\n"
            
            # Add remaining metrics
            for key, value in sorted(metrics.items()):
                if key not in priority_metrics:
                    report += f"| {key} | {value:.4f} |\n"
        else:
            report += "*No metrics logged*\n"
        
        report += "\n---\n\n## Artifacts\n\n"
        
        # List artifacts
        try:
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id)
            
            if artifacts:
                for artifact in artifacts:
                    artifact_path = artifact.path
                    if artifact.is_dir:
                        report += f"- 📁 **{artifact_path}/**\n"
                    else:
                        # Determine icon based on file type
                        if artifact_path.endswith('.png') or artifact_path.endswith('.jpg'):
                            icon = "🖼️"
                        elif artifact_path.endswith('.pt') or artifact_path.endswith('.pth'):
                            icon = "🤖"
                        elif artifact_path.endswith('.csv'):
                            icon = "📊"
                        else:
                            icon = "📄"
                        report += f"- {icon} **{artifact_path}**\n"
            else:
                report += "*No artifacts logged*\n"
        except Exception as e:
            report += f"*Error listing artifacts: {e}*\n"
        
        report += "\n---\n\n## MLflow UI Link\n\n"
        report += f"[View in MLflow]({self.tracking_uri}/#/experiments/{info.experiment_id}/runs/{run_id})\n\n"
        
        report += "---\n\n"
        report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def _create_experiment_summary_markdown(
        self,
        experiment_name: str,
        runs_df: pd.DataFrame,
        top_n: int,
    ) -> str:
        """Create markdown summary for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            runs_df: DataFrame of runs
            top_n: Number of top runs
            
        Returns:
            Markdown summary string
        """
        report = f"""# Experiment Summary: {experiment_name}

## Overview

- **Total Runs**: {len(runs_df)}
- **Top Runs Shown**: {min(top_n, len(runs_df))}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Best Performing Runs

"""
        # Create leaderboard
        report += "| Rank | Run Name | Approach | Model | Val SNR (dB) | Val Loss | Status |\n"
        report += "|------|----------|----------|-------|--------------|----------|--------|\n"
        
        for idx, row in runs_df.head(top_n).iterrows():
            rank = idx + 1
            run_name = row.get('tags.mlflow.runName', 'Unnamed')
            approach = row.get('tags.approach', 'N/A')
            model = row.get('tags.model', 'N/A')
            val_snr = row.get('metrics.val_snr', 0)
            val_loss = row.get('metrics.val_loss', 0)
            status = row.get('status', 'UNKNOWN')
            
            report += f"| {rank} | {run_name} | {approach} | {model} | **{val_snr:.2f}** | {val_loss:.4f} | {status} |\n"
        
        report += "\n---\n\n## Performance Distribution\n\n"
        
        # Statistics
        if 'metrics.val_snr' in runs_df.columns:
            snr_stats = runs_df['metrics.val_snr'].describe()
            report += "### SNR Statistics\n\n"
            report += f"- **Mean**: {snr_stats['mean']:.2f} dB\n"
            report += f"- **Std Dev**: {snr_stats['std']:.2f} dB\n"
            report += f"- **Min**: {snr_stats['min']:.2f} dB\n"
            report += f"- **Max**: {snr_stats['max']:.2f} dB\n"
            report += f"- **Median**: {snr_stats['50%']:.2f} dB\n\n"
        
        report += "---\n\n## Approach Comparison\n\n"
        
        # Compare approaches
        if 'tags.approach' in runs_df.columns and 'metrics.val_snr' in runs_df.columns:
            approach_stats = runs_df.groupby('tags.approach')['metrics.val_snr'].agg(['mean', 'std', 'max', 'count'])
            
            report += "| Approach | Avg SNR (dB) | Std Dev | Best SNR (dB) | Runs |\n"
            report += "|----------|--------------|---------|---------------|------|\n"
            
            for approach, stats in approach_stats.iterrows():
                report += f"| {approach} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['max']:.2f} | {int(stats['count'])} |\n"
        
        report += "\n---\n\n## MLflow UI Link\n\n"
        report += f"[View Experiment in MLflow]({self.tracking_uri}/#/experiments)\n\n"
        
        report += "---\n\n"
        report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def _group_parameters(self, params: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Group parameters by prefix.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Grouped parameters
        """
        groups = {}
        
        for key, value in params.items():
            if '.' in key:
                prefix = key.split('.')[0]
                param_name = '.'.join(key.split('.')[1:])
            else:
                prefix = "General"
                param_name = key
            
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][param_name] = value
        
        return groups
    
    def _format_duration(self, duration) -> str:
        """Format timedelta as human-readable string.
        
        Args:
            duration: timedelta object
            
        Returns:
            Formatted duration string
        """
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)


def generate_report_for_run(
    run_id: str,
    tracking_uri: str = "http://localhost:5050",
    reports_dir: str = "reports",
) -> str:
    """Convenience function to generate report for a run.
    
    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI
        reports_dir: Directory to save reports
        
    Returns:
        Path to generated report
    """
    generator = ExperimentReportGenerator(tracking_uri, reports_dir)
    return generator.generate_run_report(run_id)
