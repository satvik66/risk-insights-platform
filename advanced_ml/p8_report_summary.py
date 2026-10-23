"""P8: risk brief from structured metrics (templated narration, no LLM) + numeric hallucination verifier."""
import json, re, pandas as pd
from common import OUT
sc = json.load(open(f"{OUT}/p1_scorecard.json")); dr = pd.read_csv(f"{OUT}/p3_drift_report.csv")
def source_numbers():
    nums = {v for v in sc.values() if isinstance(v, (int, float))} | set(sc["importance"].values())
    nums |= set(dr.select_dtypes("number").to_numpy().ravel().tolist()) | {0.2, 0.25, 0.03, 30}   # policy constants
    return nums
def brief():
    first = dr[dr.alert != "ok"].iloc[0]; last = dr.iloc[-1]
    return (f"Out-of-time AUC is {sc['auc_test_oot']:.3f} (KS {sc['ks_test']:.3f}) versus {sc['auc_train']:.3f} in training. "
            f"Observed default rate rose from {sc['default_rate_train']:.1%} to {sc['default_rate_test']:.1%}. "
            f"Utilization PSI first breached 0.2 in month {int(first.month)} ({first.psi_utilization:.2f}) and reached {last.psi_utilization:.2f} by month {int(last.month)}. "
            f"Policy requires a documented drift investigation above 0.2; recommend retraining.")
def verify(text):
    src = source_numbers(); bad = []
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        d = len(tok.split(".")[1]) if "." in tok else 0; x = float(tok)
        if not any(round(c, d) == x for v in src for c in (v, v * 100)): bad.append(tok)
    return bad
if __name__ == "__main__":
    b = brief(); print(b, "\nunverified numbers:", verify(b))
    fake = b + " AUC fell to 0.61 after a 14.2% jump in delinquencies."
    print("\ninjected hallucination -> unverified numbers:", verify(fake))
