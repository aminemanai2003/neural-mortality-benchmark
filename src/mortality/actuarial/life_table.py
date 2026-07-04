from __future__ import annotations

import numpy as np


def mx_to_qx(mx: np.ndarray) -> np.ndarray:
    """Convert central death rates to probabilities of death (1-year ages)."""
    return 1.0 - np.exp(-mx)


def life_table(mx: np.ndarray, radix: int = 100_000) -> dict[str, np.ndarray]:
    """Build a period life table from a 1D vector of mx (ages 0..n-1)."""
    qx = mx_to_qx(mx)
    qx = np.clip(qx, 0.0, 1.0)
    n = len(qx)

    lx = np.zeros(n + 1)
    lx[0] = radix
    dx = np.zeros(n)
    Lx = np.zeros(n)

    for x in range(n):
        dx[x] = lx[x] * qx[x]
        lx[x + 1] = lx[x] - dx[x]
        Lx[x] = (lx[x] + lx[x + 1]) / 2.0

    Tx = np.zeros(n)
    Tx[n - 1] = Lx[n - 1]
    for x in range(n - 2, -1, -1):
        Tx[x] = Tx[x + 1] + Lx[x]

    ex = np.where(lx[:n] > 0, Tx / lx[:n], 0.0)

    return {"qx": qx, "lx": lx, "dx": dx, "Lx": Lx, "Tx": Tx, "ex": ex}


def life_expectancy_at(mx: np.ndarray, age: int = 0) -> float:
    """Period life expectancy at a given age."""
    lt = life_table(mx)
    if age >= len(lt["ex"]):
        return 0.0
    return float(lt["ex"][age])


def annuity_due(mx: np.ndarray, start_age: int = 65, interest_rate: float = 0.02) -> float:
    """Whole-life annuity-due ä_x at a given interest rate."""
    lt = life_table(mx)
    lx = lt["lx"]
    v = 1.0 / (1.0 + interest_rate)
    n = len(mx)

    if start_age >= n:
        return 0.0

    total = 0.0
    for t in range(n - start_age):
        age = start_age + t
        if lx[start_age] == 0:
            break
        survival = lx[age] / lx[start_age]
        total += survival * (v ** t)

    return total
