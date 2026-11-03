import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import streamlit as st, pandas as pd
st.set_page_config(page_title="Late payment risk", layout="wide")
st.title("Late payment risk model with release gates")
st.caption("Real data: UCI Default of Credit Card Clients (30,000 customers, CC BY 4.0). Closest public substitute for SMB invoices.")
r = json.loads((ROOT / "docs/results.json").read_text())
a, b, c, d = st.columns(4)
a.metric("Rule baseline AUC", r["rule_baseline_auc (PAY_0>=1)"]); b.metric("Logistic AUC", r["logistic"]["auc"]); c.metric("GBM AUC", r["gbm"]["auc"])
d.metric("Default rate (test)", f"{r['base_rate']:.1%}")
st.subheader("Release gates")
st.dataframe(pd.DataFrame([{"model": k, "auc": r[k]["auc"], "brier": r[k]["brier"], "psi_limit": r[k]["psi_limit"], "release": r[k]["gate"]["release"], "failed": ", ".join(r[k]["gate"]["failed"])} for k in ("logistic", "gbm")]
    + [{"model": "gbm on shifted batch (high limit only)", "auc": r["shifted_batch"]["gbm_auc_on_batch"], "brier": None, "psi_limit": r["shifted_batch"]["psi_limit"], "release": r["shifted_batch_gate"]["release"], "failed": ", ".join(r["shifted_batch_gate"]["failed"])}]),
    use_container_width=True, hide_index=True)
