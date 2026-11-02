import os; os.environ.setdefault("OMP_NUM_THREADS","1")
import json,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
from .data import make
NUM=["amount","terms_days","customer_age_m","prior_late_rate","is_new_customer"]; CAT=["industry"]
def pre(): return ColumnTransformer([("n",StandardScaler(),NUM),("c",OneHotEncoder(handle_unknown="ignore"),CAT)])
def psi(a,b,bins=10):
    q=np.quantile(a,np.linspace(0,1,bins+1)); q[0],q[-1]=-np.inf,np.inf
    pa=np.histogram(a,q)[0]/len(a)+1e-6; pb=np.histogram(b,q)[0]/len(b)+1e-6; return float(np.sum((pa-pb)*np.log(pa/pb)))
GATES=dict(max_auc_gap=0.02,max_brier=0.20,max_psi=0.2)
def gate(m): 
    fails=[k for k,ok in [("auc",m["auc"]>=m["oracle_auc"]-GATES["max_auc_gap"]),("brier",m["brier"]<=GATES["max_brier"]),("psi_amount",m["psi_amount"]<=GATES["max_psi"])] if not ok]
    return {"release":not fails,"failed":fails}
def run(seed=0):
    tr,te=make(20000,seed),make(8000,seed+1); X=NUM+CAT; orc=round(roc_auc_score(te.late,te.p_true),4); out={"base_rate":round(float(te.late.mean()),4),"oracle_auc_true_probabilities":orc}
    for name,est in [("logistic",LogisticRegression(max_iter=1000)),("gbm",HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05))]:
        m=make_pipeline(pre(),est).fit(tr[X],tr.late); p=m.predict_proba(te[X])[:,1]
        out[name]=dict(auc=round(roc_auc_score(te.late,p),4),brier=round(brier_score_loss(te.late,p),4),psi_amount=round(psi(tr.amount,te.amount),4),oracle_auc=orc)
        out[name]["gate"]=gate(out[name])
    drift=make(8000,seed+2,shift=0.8); out["drifted_batch_psi_amount"]=round(psi(tr.amount,drift.amount),4)
    out["drifted_batch_gate"]=gate(dict(out["gbm"],psi_amount=out["drifted_batch_psi_amount"]))
    return out
if __name__=="__main__": print(json.dumps(run(),indent=1))
