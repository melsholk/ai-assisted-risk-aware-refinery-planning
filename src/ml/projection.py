from __future__ import annotations
from typing import Dict, Tuple
import re
import numpy as np

from src.model import Data, LPParams


_VAR_X = re.compile(r"^x\[(.+)\]$")
_VAR_T = re.compile(r"^T\[(.+)\]$")
_VAR_IMP = re.compile(r"^imp_fs\[(.+),(.+)\]$")


def project_first_stage(policy, data, params):
    """
    Project ML-predicted first-stage decisions to a simple feasible set:
      - Nonnegativity
      - CDU crude total <= cap['CDU'] (scale)
      - Unit throughputs within caps, and FCC_G + FCC_D <= cap['FCC'] (scale)
      - imp_fs bounds [0, imp_max]
    This is not the full feasible region, but it enforces the key hard bounds.
    """
    out = {k: float(v) for k, v in policy.items()}

    # --- clamp nonnegativity first ---
    for k in list(out.keys()):
        if out[k] < 0.0:
            out[k] = 0.0

    # --- crude slate scaling (CDU cap) ---
    crudes = []
    for k in out.keys():
        m = _VAR_X.match(k)
        if m:
            crudes.append(k)

    cdu_cap = float(params.cap.get("CDU", np.inf))
    crude_total = float(sum(out[k] for k in crudes))
    if crude_total > cdu_cap + 1e-9 and crude_total > 1e-12:
        s = cdu_cap / crude_total
        for k in crudes:
            out[k] *= s

    # --- unit throughput clamps + FCC split scaling ---
    units = []
    for k in out.keys():
        m = _VAR_T.match(k)
        if m:
            units.append((k, m.group(1)))

    # clamp each unit to its cap (except FCC handled below)
    for k, u in units:
        cap = float(params.cap.get(u, np.inf))
        if out[k] > cap:
            out[k] = cap

    # FCC modes sum <= FCC cap
    fcc_cap = float(params.cap.get("FCC", np.inf))
    fcc_g = out.get("T[FCC_G]", 0.0)
    fcc_d = out.get("T[FCC_D]", 0.0)
    fcc_sum = float(fcc_g + fcc_d)
    if fcc_sum > fcc_cap + 1e-9 and fcc_sum > 1e-12:
        s = fcc_cap / fcc_sum
        out["T[FCC_G]"] = fcc_g * s
        out["T[FCC_D]"] = fcc_d * s

    # --- contracted imports bounds ---
    # imp_fs[p,m] <= imp_max[p,m] from params.imp_max
    for k in list(out.keys()):
        m = _VAR_IMP.match(k)
        if not m:
            continue
        p = m.group(1).strip()
        mm = m.group(2).strip()
        ub = float(params.imp_max.get((p, mm), np.inf))
        if out[k] > ub:
            out[k] = ub

    return out
