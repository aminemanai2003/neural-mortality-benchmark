import numpy as np

from mortality.actuarial.life_table import annuity_due, life_expectancy_at, life_table, mx_to_qx


def test_mx_to_qx_zero():
    assert mx_to_qx(np.array([0.0]))[0] == 0.0


def test_mx_to_qx_large():
    qx = mx_to_qx(np.array([10.0]))[0]
    assert 0.99 < qx <= 1.0


def test_life_table_shapes():
    mx = np.full(101, 0.01)
    lt = life_table(mx)
    assert lt["qx"].shape == (101,)
    assert lt["lx"].shape == (102,)
    assert lt["ex"].shape == (101,)


def test_e0_reasonable():
    mx = np.full(101, 0.01)
    e0 = life_expectancy_at(mx, age=0)
    assert 40 < e0 < 100


def test_annuity_positive():
    mx = np.full(101, 0.01)
    a65 = annuity_due(mx, start_age=65)
    assert a65 > 0
