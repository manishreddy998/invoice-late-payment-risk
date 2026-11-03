import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from .data import load, split, NUM


def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-6; pb = np.histogram(b, q)[0] / len(b) + 1e-6; return float(np.sum((pa - pb) * np.log(pa / pb)))


# Real data has no known ceiling, so the model must beat the current simple rule by a margin.
GATES = dict(min_auc_gain_over_rule=0.02, max_brier=0.20, max_psi=0.2)


def gate(m):
    fails = [k for k, ok in [("auc", m["auc"] >= m["rule_auc"] + GATES["min_auc_gain_over_rule"]), ("brier", m["brier"] <= GATES["max_brier"]),
                             ("psi_limit", m["psi_limit"] <= GATES["max_psi"])] if not ok]
    return {"release": not fails, "failed": fails}


def run(seed=0):
    df = load(); tr, te = split(df, seed)
    rule = (te.PAY_0 >= 1).astype(float) + 0.1 * te.PAY_2.clip(0)  # baseline: late last month, then by months late
    out = {"rows_train": len(tr), "rows_test": len(te), "base_rate": round(float(te.late.mean()), 4),
           "rule_baseline_auc (PAY_0>=1)": round(roc_auc_score(te.late, rule), 4)}
    ra = out["rule_baseline_auc (PAY_0>=1)"]
    for name, est in [("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                      ("gbm", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0))]:
        m = est.fit(tr[NUM], tr.late); p = m.predict_proba(te[NUM])[:, 1]
        out[name] = dict(auc=round(roc_auc_score(te.late, p), 4), brier=round(brier_score_loss(te.late, p), 4),
                         psi_limit=round(psi(tr.LIMIT_BAL, te.LIMIT_BAL), 4), rule_auc=ra)
        out[name]["gate"] = gate(out[name])
    # Real subgroup shift: a batch of only high limit customers (top quartile), as if a new segment onboarded.
    hi = te[te.LIMIT_BAL >= tr.LIMIT_BAL.quantile(0.75)]
    p = est.predict_proba(hi[NUM])[:, 1]
    out["shifted_batch"] = {"rows": len(hi), "psi_limit": round(psi(tr.LIMIT_BAL, hi.LIMIT_BAL), 4), "late_rate": round(float(hi.late.mean()), 4),
                            "gbm_auc_on_batch": round(roc_auc_score(hi.late, p), 4)}
    out["shifted_batch_gate"] = gate(dict(out["gbm"], psi_limit=out["shifted_batch"]["psi_limit"]))
    return out


if __name__ == "__main__":
    r = run(); Path("docs").mkdir(exist_ok=True); Path("docs/results.json").write_text(json.dumps(r, indent=1)); print(json.dumps(r, indent=1))
