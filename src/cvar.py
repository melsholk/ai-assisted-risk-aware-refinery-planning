from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from scipy.optimize import linprog

from src.model import Data, build_lp_matrices
from src.scenarios import Scenario


@dataclass(frozen=True)
class CVaRResult:
    success: bool
    message: str
    alpha: float
    lam: float
    objective: float
    first_stage: Dict[str, float]
    scenario_loss: Dict[str, float]
    z: float
    expected_loss: float
    cvar_loss: float


def _weighted_var_cvar_loss(losses, probs, alpha):
    """
    Compute VaR_alpha(loss) and CVaR_alpha(loss) for a discrete distribution.
    losses: shape (S,)
    probs: shape (S,), sums to 1
    """
    order = np.argsort(losses)
    L = losses[order]
    p = probs[order]
    cdf = np.cumsum(p)

    # VaR: smallest loss where CDF >= alpha
    k = int(np.searchsorted(cdf, alpha, side="left"))
    k = min(max(k, 0), len(L) - 1)
    var = float(L[k])

    # CVaR: tail mean over losses >= VaR (discrete approximation)
    tail = losses >= var - 1e-12
    tail_prob = float(np.sum(probs[tail]))
    cvar = float(np.sum(probs[tail] * losses[tail]) / max(tail_prob, 1e-12))
    return var, cvar


def solve_cvar_extensive_form(
    data,
    scenarios,
    alpha = 0.90,
    lam = 1.0,
    first_stage_vars = None,
):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if len(scenarios) == 0:
        raise ValueError("Need at least one scenario")

    # Normalize probabilities
    probs = np.array([sc.prob for sc in scenarios], dtype=float)
    probs = probs / probs.sum()

    # Build per-scenario LP templates
    lp_list = [build_lp_matrices(data, params=sc.params) for sc in scenarios]
    n0 = len(lp_list[0]["var_names"])
    names0 = lp_list[0]["var_names"]

    for k, lp in enumerate(lp_list):
        if lp["var_names"] != names0:
            raise RuntimeError(
                f"Scenario {scenarios[k].name} has different variable ordering. "
                "Make sure build_lp_matrices always adds vars in a stable order."
            )

    if first_stage_vars is None:
        crudes = lp_list[0]["meta"]["crudes"]
        first_stage_vars = [f"x[{c}]" for c in crudes] + [
            "T[FCC_G]", "T[FCC_D]", "T[VR]", "T[REF]", "T[HC]", "T[HDT]"
        ]

    idx0 = lp_list[0]["idx"]
    fs_idx = []
    for v in first_stage_vars:
        if v not in idx0:
            raise ValueError(f"First-stage var '{v}' not found in LP variables.")
        fs_idx.append(idx0[v])

    S = len(scenarios)

    # Extra vars: z + s_omega for each scenario
    z_pos = S * n0
    s_pos = [S * n0 + 1 + i for i in range(S)]
    N = S * n0 + 1 + S

    # Objective: expected loss + lam * CVaR
    c = np.zeros(N, dtype=float)
    for i, lp in enumerate(lp_list):
        c[i * n0:(i + 1) * n0] = probs[i] * lp["c"]

    # CVaR terms only matter when lam>0
    c[z_pos] = lam * 1.0
    for i in range(S):
        c[s_pos[i]] = lam * (probs[i] / (1.0 - alpha))

    # Bounds
    bounds = []
    for i in range(S):
        bounds.extend(lp_list[i]["bounds"])
    bounds.append((None, None))  # z free
    for _ in range(S):
        bounds.append((0.0, None))  # s_i >= 0

    # Constraints
    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    # 1) Scenario constraints (block diagonal)
    for i, lp in enumerate(lp_list):
        off = i * n0
        if lp["A_eq"] is not None:
            for r, rhs in zip(lp["A_eq"], lp["b_eq"]):
                row = np.zeros(N)
                row[off:off + n0] = r
                A_eq.append(row)
                b_eq.append(float(rhs))

        if lp["A_ub"] is not None:
            for r, rhs in zip(lp["A_ub"], lp["b_ub"]):
                row = np.zeros(N)
                row[off:off + n0] = r
                A_ub.append(row)
                b_ub.append(float(rhs))

    # 2) Non-anticipativity
    for j in fs_idx:
        for i in range(1, S):
            row = np.zeros(N)
            row[i * n0 + j] = 1.0
            row[0 * n0 + j] = -1.0
            A_eq.append(row)
            b_eq.append(0.0)

    # 3) Linking constraints: L_i - z - s_i <= 0
    for i, lp in enumerate(lp_list):
        row = np.zeros(N)
        row[i * n0:(i + 1) * n0] = lp["c"]
        row[z_pos] = -1.0
        row[s_pos[i]] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    res = linprog(
        c=c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return CVaRResult(
            success=False,
            message=res.message,
            alpha=alpha,
            lam=lam,
            objective=float(res.fun) if res.fun is not None else float("nan"),
            first_stage={},
            scenario_loss={},
            z=float("nan"),
            expected_loss=float("nan"),
            cvar_loss=float("nan"),
        )

    x = res.x

    # First-stage decisions from scenario 0 block
    first_stage = {v: float(x[idx0[v]]) for v in first_stage_vars}

    # Scenario losses at optimum
    losses = np.zeros(S, dtype=float)
    loss_by_name: Dict[str, float] = {}
    for i, sc in enumerate(scenarios):
        xi = x[i * n0:(i + 1) * n0]
        Li = float(lp_list[i]["c"] @ xi)
        losses[i] = Li
        loss_by_name[sc.name] = Li

    expected_loss = float(np.sum(probs * losses))

    if lam > 0:
        # Use solver z and s (they are meaningful when penalized)
        z = float(x[z_pos])
        s = np.array([float(x[pos]) for pos in s_pos], dtype=float)
        cvar_loss = float(z + (1.0 / (1.0 - alpha)) * np.sum(probs * s))
    else:
        # When lam=0, z and s are not uniquely determined; compute VaR/CVaR from losses.
        z, cvar_loss = _weighted_var_cvar_loss(losses, probs, alpha)

    return CVaRResult(
        success=True,
        message=res.message,
        alpha=alpha,
        lam=lam,
        objective=float(res.fun),
        first_stage=first_stage,
        scenario_loss=loss_by_name,
        z=z,
        expected_loss=expected_loss,
        cvar_loss=cvar_loss,
    )
