"""
Helper script to generate experiment reports from MLflow.

Run with local .venv environment:
    .\.venv\Scripts\Activate.ps1
    python scripts/generate_report.py --run-id <run_id>
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ecg_digitization.utils import ExperimentReportGenerator, generate_report_for_run
import mlflow


def main():
    parser = argparse.ArgumentParser(description="Generate experiment reports from MLflow")
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run ID to generate report for",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Generate report for latest run",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate experiment summary (top 10 runs)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ecg-digitization",
        help="Experiment name (default: ecg-digitization)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5050",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top runs to include in summary",
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ExperimentReportGenerator(
        tracking_uri=args.tracking_uri,
        reports_dir="reports",
    )
    
    if args.summary:
        # Generate experiment summary
        print(f"Generating summary for experiment: {args.experiment}")
        report_path = generator.generate_experiment_summary(
            experiment_name=args.experiment,
            top_n=args.top_n,
        )
        print(f"✓ Summary report: {report_path}")
    
    elif args.latest:
        # Get latest run
        mlflow.set_tracking_uri(args.tracking_uri)
        experiment = mlflow.get_experiment_by_name(args.experiment)
        
        if experiment is None:
            print(f"✗ Experiment '{args.experiment}' not found")
            return 1
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1,
            order_by=["start_time DESC"],
        )
        
        if len(runs) == 0:
            print(f"✗ No runs found in experiment '{args.experiment}'")
            return 1
        
        run_id = runs.iloc[0]['run_id']
        print(f"Latest run: {run_id}")
        
        report_path = generate_report_for_run(
            run_id=run_id,
            tracking_uri=args.tracking_uri,
        )
        print(f"✓ Run report: {report_path}")
    
    elif args.run_id:
        # Generate report for specific run
        print(f"Generating report for run: {args.run_id}")
        report_path = generate_report_for_run(
            run_id=args.run_id,
            tracking_uri=args.tracking_uri,
        )
        print(f"✓ Run report: {report_path}")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
