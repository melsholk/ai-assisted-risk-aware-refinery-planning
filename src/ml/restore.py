from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog

from src.model import Data, LPParams, build_lp_matrices


def restore_first_stage(
    data,
    params,
    scenarios_params,
    first_stage_vars,
    yhat,
    w = 1.0,
    slack_penalty = 1e6,
):
    """
    Find a feasible first-stage policy close to ML prediction yhat.

    We solve a linear program:
        minimize  sum_i w_i * (u_i + v_i) + slack_penalty * sum slack
        s.t.      x_i - yhat_i = u_i - v_i
                  u_i, v_i >= 0
                  (x, second-stage vars) satisfy deterministic LP constraints for scenarios_params
                  x_i are the first-stage vars

    Implementation trick:
    - Build deterministic LP for ONE representative params set (we use scenarios_params).
    - Add deviation variables u,v for each first-stage var and tie them to x_i.
    - Add artificial slacks to constraints if needed (optional; we’ll start without).
    """
    lp = build_lp_matrices(data, params=scenarios_params)
    idx = lp["idx"]
    nvar = len(lp["c"])

    # Prepare extra variables: u_i, v_i for each fs var
    fs_idx = [idx[nm] for nm in first_stage_vars if nm in idx]
    m = len(fs_idx)

    # Decision vector: [orig_vars (nvar), u(m), v(m)]
    N = nvar + 2 * m

    # Objective: original objective is loss; we don't want to optimize economics here,
    # only feasibility + closeness. So set original c to zeros.
    c = np.zeros(N, dtype=float)
    c[nvar:nvar+m] = w
    c[nvar+m:nvar+2*m] = w

    # Bounds: original bounds + u/v >= 0
    bounds = list(lp["bounds"]) + [(0.0, None)] * (2 * m)

    # Constraints: original A_ub/A_eq extended with zeros for u/v
    def extend_A(A):
        if A is None:
            return None
        A = np.asarray(A)
        extra = np.zeros((A.shape[0], 2 * m), dtype=float)
        return np.hstack([A, extra])

    A_ub = extend_A(lp["A_ub"])
    b_ub = lp["b_ub"]
    A_eq = extend_A(lp["A_eq"])
    b_eq = lp["b_eq"]

    # Add equations tying x_i to yhat_i via u_i - v_i:
    # x_i - u_i + v_i = yhat_i
    A_tie = np.zeros((m, N), dtype=float)
    b_tie = np.zeros(m, dtype=float)
    for j, orig_j in enumerate(fs_idx):
        A_tie[j, orig_j] = 1.0
        A_tie[j, nvar + j] = -1.0       # -u
        A_tie[j, nvar + m + j] = 1.0    # +v
        b_tie[j] = float(yhat.get(first_stage_vars[j], 0.0))

    if A_eq is None:
        A_eq2 = A_tie
        b_eq2 = b_tie
    else:
        A_eq2 = np.vstack([A_eq, A_tie])
        b_eq2 = np.concatenate([b_eq, b_tie])

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq2,
        b_eq=b_eq2,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        # fallback: return clipped yhat (last resort)
        return {nm: max(0.0, float(yhat.get(nm, 0.0))) for nm in first_stage_vars}

    x = res.x
    policy = {nm: float(x[idx[nm]]) for nm in first_stage_vars if nm in idx}
    return policy
