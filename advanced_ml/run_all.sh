set -e; cd "$(dirname "$0")"; mkdir -p outputs
for p in p1_credit_default p2_fraud p3_drift_monitor p4_segmentation p5_feature_pipeline p6_rag p7_nl2sql p8_report_summary p9_hybrid_search p10_agent capstone; do echo "== $p"; python3 $p.py; done
