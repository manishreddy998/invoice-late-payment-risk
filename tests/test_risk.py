import numpy as np
from risk.train import psi,gate,run
from risk.data import make
def test_psi_zero_same(): a=np.arange(1000.); assert psi(a,a)<1e-3
def test_data_seeded(): assert make(10,1).equals(make(10,1))
def test_gate_blocks_drift():
    r=run(); assert r["gbm"]["gate"]["release"] and not r["drifted_batch_gate"]["release"]
def test_gate_logic(): assert gate(dict(auc=.6,oracle_auc=.7,brier=.1,psi_amount=0))["failed"]==["auc"]
