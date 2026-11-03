import numpy as np
from risk.train import psi, gate, run
from risk.data import load


def test_psi_zero_same(): a = np.arange(1000.); assert psi(a, a) < 1e-3
def test_real_data_shape(): d = load(); assert len(d) == 30000 and set(d.late.unique()) == {0, 1}
def test_gate_logic(): assert gate(dict(auc=.6, rule_auc=.7, brier=.1, psi_limit=0))["failed"] == ["auc"]
def test_shift_gate_blocks():
    r = run(); assert not r["shifted_batch_gate"]["release"]
