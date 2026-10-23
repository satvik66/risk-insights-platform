DOCS = {
 "credit_policy": "Maximum debt-to-income ratio for unsecured card approvals is 45 percent. Exceptions require senior underwriter sign-off. Applicants above 50 percent DTI are declined automatically.",
 "utilization_policy": "Revolving utilization above 80 percent triggers an account review. Utilization trends are monitored monthly. A sustained increase across the portfolio is escalated to the Credit Risk Committee.",
 "model_monitoring": "Population Stability Index above 0.2 on any model input requires a documented drift investigation. A PSI above 0.25 on the score distribution mandates retraining within 30 days.",
 "model_validation": "All credit models must be independently validated before production. Validation covers discriminatory power measured by AUC and KS, calibration, and stability on out of time samples.",
 "fraud_ops": "Transactions scored above the fraud threshold are routed to the review queue. Analysts must disposition each alert within 24 hours. Alerts on new devices with high velocity are prioritized.",
 "data_retention": "Customer transaction data is retained for seven years. Personal data must be masked in non production environments. Access to customer records is logged.",
 "incident_response": "Security incidents must be reported to the on-call security lead within one hour of detection. A post incident review is held within five business days.",
 "collections": "Accounts that are 30 days past due enter early stage collections. Hardship plans may extend payment terms by up to twelve months. Extensions require manager approval.",
 "ai_governance": "Generative AI assistants may only use approved internal data sources. Outputs that cite figures must be traceable to source systems. Human approval is required before any external communication.",
 "kubernetes_ops": "Production workloads on OpenShift must define resource requests and limits and readiness probes. Containers run as non root. Rollouts use canary deployment with automatic rollback on error rate increase.",
 "provisioning": "Loan loss provisions are recalculated quarterly using expected credit loss. Stage 2 classification applies when credit risk has increased significantly since origination. Delinquency over 30 days is such a trigger.",
 "feature_governance": "Features used in production models must be registered with an owner and definition. Point in time correctness is required. No information from after the prediction date may be used in training.",
}
GOLDEN = [
 ("What DTI limit applies for credit card approval?", "credit_policy"),
 ("When must a model be retrained because of drift?", "model_monitoring"),
 ("how fast must analysts handle fraud alerts", "fraud_ops"),
 ("who can approve longer repayment terms for struggling customers", "collections"),
 ("How long do we keep customer records", "data_retention"),
 ("rules for chatbots that use company data", "ai_governance"),
 ("what happens when borrowers max out their cards", "utilization_policy"),
 ("how quickly must a breach be reported", "incident_response"),
 ("containers must not run as root", "kubernetes_ops"),
 ("can training features leak future information", "feature_governance"),
 ("independent review of scoring models before go live", "model_validation"),
 ("when do we reclassify loans to stage 2", "provisioning"),
]
OUT_OF_SCOPE = ["what is the weather in Dallas tomorrow", "best pizza recipe for a party"]
