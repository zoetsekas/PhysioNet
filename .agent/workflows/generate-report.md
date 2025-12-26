---
description: Local generate experiment report from MLflow run
---

# Generate Experiment Report

Creates a comprehensive markdown report for a specific MLflow run using the local Python environment.

## Usage

### 1. Activate Local Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Generate Report for Latest Run

```powershell
python -c "
import mlflow
from ecg_digitization.utils import generate_report_for_run

# Get latest run
mlflow.set_tracking_uri('http://localhost:5050')
experiment = mlflow.get_experiment_by_name('ecg-digitization')
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
run_id = runs.iloc[0]['run_id']

# Generate report
report_path = generate_report_for_run(run_id)
print(f'Report: {report_path}')
"
```

### 3. Generate Report for Specific Run ID

```powershell
python -c "
from ecg_digitization.utils import generate_report_for_run
generate_report_for_run('YOUR_RUN_ID_HERE')
"
```

### 4. Generate Experiment Summary

```powershell
python -c "
from ecg_digitization.utils import ExperimentReportGenerator
generator = ExperimentReportGenerator()
generator.generate_experiment_summary('ecg-digitization', top_n=10)
"
```

## When to Use

- After training completes to review results
- To compare multiple runs offline
- To share experiment results without MLflow access
- For documentation and version control

## Output

Reports saved to `reports/` directory:
- `run_{run_id}_{timestamp}.md` - Individual run reports
- `experiment_{name}_{timestamp}.md` - Summary reports