# Experiment Reports

This directory contains automatically generated experiment reports from MLflow runs.

## Report Types

### 1. Run Reports
Individual experiment run reports with detailed information:
- Configuration parameters
- Performance metrics
- Training artifacts
- Links to MLflow UI

**Format**: `run_{run_id}_{timestamp}.md`

### 2. Experiment Summaries
Comparative reports across multiple runs:
- Leaderboard of top performers
- Approach comparisons
- Performance statistics

**Format**: `experiment_{name}_{timestamp}.md`

## Automatic Generation

Reports are automatically generated after each training run completes.

## Manual Generation

### Generate Report for Specific Run

```python
from ecg_digitization.utils import generate_report_for_run

report_path = generate_report_for_run(
    run_id="abc123...",
    tracking_uri="http://localhost:5050",
    reports_dir="reports",
)
print(f"Report saved to: {report_path}")
```

### Generate Experiment Summary

```python
from ecg_digitization.utils import ExperimentReportGenerator

generator = ExperimentReportGenerator(
    tracking_uri="http://localhost:5050",
    reports_dir="reports",
)

summary_path = generator.generate_experiment_summary(
    experiment_name="ecg-digitization",
    top_n=10,
)
print(f"Summary saved to: {summary_path}")
```

## Report Contents

### Run Report Includes:
- ✅ Run metadata (ID, name, status, duration)
- ✅ All configuration parameters (grouped by category)
- ✅ Performance metrics (train/val loss, SNR)
- ✅ Tagged information (approach, model, etc.)
- ✅ List of logged artifacts
- ✅ Direct link to MLflow UI

### Experiment Summary Includes:
- ✅ Leaderboard of top N runs
- ✅ Performance statistics (mean, std, min, max)
- ✅ Approach comparison table
- ✅ Model comparison table
- ✅ Links to MLflow UI

## Example Usage

```bash
# Train model (report auto-generated)
python -m ecg_digitization.train approach=signalsavants

# Check reports directory
ls reports/

# View latest report
cat reports/run_abc12345_20250126_120000.md
```

## Integration with MLflow

Reports complement the MLflow UI by providing:
- **Portable Documentation**: Markdown files can be shared without MLflow access
- **Version Control**: Reports can be committed to git for historical tracking
- **Offline Access**: View experiment results without running MLflow server
- **Custom Insights**: Tailored summaries and comparisons

## Report Organization

Recommended structure:
```
reports/
├── README.md                                    # This file
├── run_abc12345_20240126_120000.md             # Individual run reports
├── run_def67890_20240126_130000.md
├── experiment_ecg-digitization_20240126.md     # Experiment summaries
└── archive/                                     # Older reports
    └── 2024-01/
```

---

**Reports are automatically generated and saved here after each experiment run.**
