"""Optional local MLflow experiment logger for saved demo metrics."""
from pathlib import Path
import json

def log_local_metrics(out='artifacts'):
    import mlflow
    root=Path(out)
    mlflow.set_experiment('synthetic_risk_platform')
    with mlflow.start_run(run_name='credit-synthetic'):
        metrics=json.loads((root/'credit_metrics.json').read_text())
        mlflow.log_metrics({k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(root/'credit_metrics.json'))
    with mlflow.start_run(run_name='fraud-synthetic'):
        metrics=json.loads((root/'fraud_metrics.json').read_text())
        mlflow.log_metrics({k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))})
        mlflow.log_artifact(str(root/'fraud_metrics.json'))
if __name__=='__main__':
    log_local_metrics()
    print('Logged demo metrics to local MLflow tracking.')
