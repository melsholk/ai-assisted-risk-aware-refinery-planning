from __future__ import annotations

import argparse
import numpy as np

from src.model import load_data, default_params, build_and_solve_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    data = load_data(args.data_dir)
    params = default_params(data)

    # Solve + print standard outputs
    out = build_and_solve_lp(data, params=params, verbose=True)

    # Consistency checks
    tol = float(args.tol)
    prod = out["products"].copy()

    # Determine export market column prefix (assumes only one export market or uses first found)
    cols = list(prod.columns)

    # Identify market prefixes from columns like "<M>_delivered_kbpd"
    market_prefixes = set()
    for c in cols:
        if c.endswith("_delivered_kbpd"):
            market_prefixes.add(c.replace("_delivered_kbpd", ""))
    market_prefixes = sorted(market_prefixes)

    local = "Local" if "Local" in market_prefixes else (market_prefixes[0] if market_prefixes else None)
    export_markets = [m for m in market_prefixes if m != local] if local is not None else []

    print("\n=== CONSISTENCY REPORT ===")
    print(f"Tolerance: {tol:g}")

    # [1] Refinery allocation: Q[p] == sum_m q_ref[p,m]
    # We only have per-market q_ref columns; sum them and compare to refinery_kbpd.
    diffs = []
    for _, r in prod.iterrows():
        q_ref_sum = 0.0
        for m in market_prefixes:
            k = f"{m}_refinery_kbpd"
            if k in prod.columns:
                q_ref_sum += float(r[k])
        diffs.append(float(r["refinery_kbpd"]) - q_ref_sum)

    max_abs = float(np.max(np.abs(diffs))) if diffs else 0.0
    print("\n[1] Refinery allocation: Q[p] == sum_m q_ref[p,m]")
    print(f"Max |diff|: {max_abs:.6g}")
    print("OK (no violations)" if max_abs <= tol else "FAIL (violations found)")

    # [2] Delivered identity: q[p,m] == q_ref[p,m] + imp_fs[p,m] + imp_spot[p,m]
    diffs2 = []
    for _, r in prod.iterrows():
        for m in market_prefixes:
            q = float(r.get(f"{m}_delivered_kbpd", 0.0))
            q_ref = float(r.get(f"{m}_refinery_kbpd", 0.0))
            imp_fs = float(r.get(f"{m}_import_fs_kbpd", 0.0))
            imp_sp = float(r.get(f"{m}_import_spot_kbpd", 0.0))
            diffs2.append(q - (q_ref + imp_fs + imp_sp))

    max_abs2 = float(np.max(np.abs(diffs2))) if diffs2 else 0.0
    print("\n[2] Delivered identity: q[p,m] == q_ref[p,m] + imp_fs[p,m] + imp_spot[p,m]")
    print(f"Max |diff|: {max_abs2:.6g}")
    print("OK (no violations)" if max_abs2 <= tol else "FAIL (violations found)")

    # [3] Local hard mins: Local_delivered >= Local_dmin
    if local is not None:
        dmin = params.dmin  # Dict[(p,m)] -> demand_min
        slack = []
        for _, r in prod.iterrows():
            p = str(r["product"])
            delivered = float(r.get(f"{local}_delivered_kbpd", 0.0))
            req = float(dmin.get((p, local), 0.0))
            slack.append(delivered - req)
        min_slack = float(np.min(slack)) if slack else 0.0

        print("\n[3] Local hard mins: Local_delivered >= Local_dmin")
        print(f"Min slack (q-dmin): {min_slack:.6g}")
        print("OK (no violations)" if min_slack >= -tol else "FAIL (violations found)")
    else:
        print("\n[3] Local hard mins: skipped (no Local market columns found)")


if __name__ == "__main__":
    main()
