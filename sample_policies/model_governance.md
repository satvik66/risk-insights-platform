# Synthetic Model Governance Standard
DEMONSTRATION ONLY. Not an official policy.

## Monitoring cadence
Model owners review model performance and drift on a scheduled basis. Drift signals include PSI and changes in feature distributions. Material drift prompts investigation, but PSI alone does not establish model failure.

## Release requirements
Model version, training data lineage, holdout performance, known limitations, and rollback method should be recorded before promotion. Significant changes require a human approval step. Post-deployment monitoring includes prediction distribution, latency, failures, and model performance when labels become available.

## Privacy and safety
Do not enter actual account numbers, names, personal identifiers, confidential banking policies, or secrets into a public demo. Outputs are informational and require human review.
