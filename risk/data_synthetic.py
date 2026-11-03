"""Synthetic small business invoices with a known generating process for late payment (60+ days)."""
import numpy as np,pandas as pd
def make(n=20000,seed=0,shift=0.0):
    r=np.random.default_rng(seed)
    df=pd.DataFrame(dict(amount=np.round(r.lognormal(7+shift,1.0,n),2),terms_days=r.choice([15,30,45,60],n),customer_age_m=r.integers(0,120,n),
        prior_late_rate=r.beta(1.5,6,n),industry=r.choice(["retail","construction","services","food","tech"],n),is_new_customer=r.random(n)<0.2))
    ind={"retail":0,"construction":.7,"services":.1,"food":.4,"tech":-.2}
    z=-2.6+0.35*np.log(df.amount/1000+1)+0.012*(df.terms_days-30)-0.008*df.customer_age_m+4.0*df.prior_late_rate+df.industry.map(ind)+0.6*df.is_new_customer
    df["p_true"]=1/(1+np.exp(-z)); df["late"]=(r.random(n)<df.p_true).astype(int); return df
