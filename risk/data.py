"""REAL data: "Default of Credit Card Clients" (Yeh & Lien 2009, UCI ML Repository id 350,
CC BY 4.0; CSV from huggingface.co/datasets/scikit-learn/credit-card-clients).
30,000 real card holders in Taiwan, 2005: credit limit, age, 6 months of repayment status
(months late), bill and payment amounts, and whether they defaulted next month.
Closest public substitute for small business invoice late payment: no public dataset of real
SMB invoices with payment outcomes and a clear licence was found. SEX, MARRIAGE and EDUCATION
are dropped from features on purpose."""
from pathlib import Path
import pandas as pd

F = Path(__file__).resolve().parent.parent / "data" / "UCI_Credit_Card.csv"
NUM = ["LIMIT_BAL", "AGE"] + [f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


def load():
    df = pd.read_csv(F).rename(columns={"default.payment.next.month": "late"})
    return df


def split(df, seed=0, frac=0.7):
    tr = df.sample(frac=frac, random_state=seed); return tr, df.drop(tr.index)
