"""P7: NL->SQL with guardrails. The translator is a deterministic grammar (no LLM available here);
validate_sql() + read-only execution + audit log are the reusable guardrail layer for LLM-generated SQL."""
import re, json, time, sqlite3, os, joblib, pandas as pd
from common import *
DB = f"{OUT}/risk.db"; COLS = {"loan_id","month","portfolio","income","utilization","dti","delinq_2y","defaulted","pd_score","loan_amt"}
KW = {"select","from","where","group","by","order","limit","avg","count","sum","round","and","or","as","desc","asc","loans","value","n","max","min"}
def build_db():
    df = make_loans().rename(columns={"default": "defaulted"}); m = joblib.load(f"{OUT}/credit_model.joblib"); df["pd_score"] = m.predict_proba(design(df.rename(columns={"defaulted":"default"})))[:, 1]
    if os.path.exists(DB): os.remove(DB)
    c = sqlite3.connect(DB); df.to_sql("loans", c, index=False); c.close()
def validate_sql(sql):
    s = sql.strip().rstrip(";")
    if ";" in s: return "multiple statements"
    if not re.match(r"(?is)^select\b", s): return "only SELECT allowed"
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b|--|/\*", s): return "forbidden keyword"
    ids = set(re.findall(r"[a-z_][a-z_0-9]*", re.sub(r"'[^']*'", "", s.lower())))
    bad = ids - KW - COLS
    return f"unknown identifiers: {sorted(bad)}" if bad else None
def translate(q):
    q = q.lower()
    metric = ("COUNT(*)" if re.search(r"how many|count|number of", q) else "ROUND(AVG(defaulted),4)" if "default rate" in q else "ROUND(AVG(pd_score),4)" if re.search(r"average pd|risk score|expected default|predicted", q)
              else "ROUND(AVG(utilization),4)" if "utilization" in q else "ROUND(AVG(dti),4)" if "dti" in q else "COUNT(*)")
    where = []
    pf = re.findall(r"\b(auto|home|card)\b", q); mo = re.search(r"month (\d+)", q)
    if pf and not re.search(r"by portfolio|per portfolio", q): where.append(f"portfolio = '{pf[0]}'")
    if mo: where.append(f"month = {int(mo.group(1))}")
    if "high utilization" in q: where.append("utilization > 0.8")
    grp = "portfolio" if re.search(r"by portfolio|per portfolio", q) else "month" if re.search(r"by month|per month|over time", q) else None
    sql = f"SELECT {(grp + ', ') if grp else ''}{metric} AS value FROM loans" + (" WHERE " + " AND ".join(where) if where else "") + (f" GROUP BY {grp} ORDER BY {grp}" if grp else "")
    return sql
def ask(q, log=f"{OUT}/p7_audit.jsonl"):
    sql = translate(q); err = validate_sql(sql)
    if not err and " limit " not in sql.lower(): sql += " LIMIT 100"
    res = None
    if not err:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); res = pd.read_sql(sql, c); c.close()
    open(log, "a").write(json.dumps(dict(ts=time.time(), q=q, sql=sql, error=err, rows=None if res is None else len(res))) + "\n")
    return sql, err, res
if __name__ == "__main__":
    open(f"{OUT}/p7_audit.jsonl", "w").close(); build_db()
    for q in ["What is the default rate by portfolio?", "average utilization in month 11 for card loans", "how many loans have high utilization in month 10", "average risk score by month"]:
        sql, err, res = ask(q); print(f"\nQ: {q}\nSQL: {sql}\n{res.head(4).to_string(index=False) if res is not None else err}")
    print("\n-- guardrail tests (direct SQL, as an LLM might emit) --")
    for bad in ["DROP TABLE loans", "SELECT * FROM loans; DELETE FROM loans", "SELECT ssn FROM loans", "SELECT 1 FROM loans -- x", "UPDATE loans SET defaulted=0"]:
        print(f"{bad!r:48} -> {validate_sql(bad)}")
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try: c.execute("DELETE FROM loans")
    except Exception as e: print("read-only connection blocks writes ->", e)
