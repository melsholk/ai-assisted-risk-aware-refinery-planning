from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Allow running as a script from repo root or elsewhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import load_data, build_and_solve_lp  # noqa: E402


def _consistency_report(data, results, tol=1e-6):
    """
    Prints consistency checks:
      1) For each product p: refinery_kbpd == sum_m refinery_delivered(p,m)
      2) For each (p,m): delivered == refinery + import
      3) For each product p: Local delivered >= Local demand_min
    """
    if not isinstance(results, dict) or "products" not in results:
        print("\n[Consistency] Skipped: build_and_solve_lp() did not return a results dict with 'products'.")
        return

    prod_df: pd.DataFrame = results["products"].copy()

    # Infer market columns from the dataframe
    delivered_cols = [c for c in prod_df.columns if c.endswith("_delivered_kbpd")]
    refinery_cols = [c for c in prod_df.columns if c.endswith("_refinery_kbpd")]
    import_cols = [c for c in prod_df.columns if c.endswith("_import_kbpd")]

    if "product" not in prod_df.columns or "refinery_kbpd" not in prod_df.columns:
        print("\n[Consistency] Skipped: 'products' table missing required columns.")
        return

    # Helper to align cols by market prefix (e.g., "Local_", "Export_")
    def market_prefix(colname: str) -> str:
        # "Local_delivered_kbpd" -> "Local"
        return colname.replace("_delivered_kbpd", "").replace("_refinery_kbpd", "").replace("_import_kbpd", "")

    markets_from_df = sorted({market_prefix(c) for c in delivered_cols} | {market_prefix(c) for c in refinery_cols})

    # 1) Refinery allocation check: refinery_kbpd == sum refinery delivered across markets
    alloc_rows = []
    for _, r in prod_df.iterrows():
        p = r["product"]
        refinery_total = float(r["refinery_kbpd"])
        sum_ref = float(np.nansum([r.get(f"{m}_refinery_kbpd", 0.0) for m in markets_from_df]))
        diff = sum_ref - refinery_total
        alloc_rows.append([p, refinery_total, sum_ref, diff])

    alloc_chk = pd.DataFrame(alloc_rows, columns=["product", "Q_refinery", "sum_q_ref", "diff(sum_q_ref - Q)"])
    alloc_max = float(np.max(np.abs(alloc_chk["diff(sum_q_ref - Q)"].values))) if len(alloc_chk) else 0.0

    # 2) Delivered identity check: delivered == refinery + import (per market)
    deliv_rows = []
    for _, r in prod_df.iterrows():
        p = r["product"]
        for m in markets_from_df:
            q = float(r.get(f"{m}_delivered_kbpd", 0.0))
            qref = float(r.get(f"{m}_refinery_kbpd", 0.0))
            qimp = float(r.get(f"{m}_import_kbpd", 0.0))
            diff = q - (qref + qimp)
            deliv_rows.append([p, m, q, qref, qimp, diff])

    deliv_chk = pd.DataFrame(
        deliv_rows,
        columns=["product", "market", "delivered", "refinery", "import", "diff(delivered-(refinery+import))"],
    )
    deliv_max = float(np.max(np.abs(deliv_chk["diff(delivered-(refinery+import))"].values))) if len(deliv_chk) else 0.0

    # 3) Local demand_min check: Local_delivered >= dmin (if Local exists)
    dmin_map = data.markets.set_index(["product", "market"])["demand_min"].to_dict()
    local_rows = []
    if "Local_delivered_kbpd" in prod_df.columns:
        for _, r in prod_df.iterrows():
            p = r["product"]
            qloc = float(r["Local_delivered_kbpd"])
            dmin = float(dmin_map.get((p, "Local"), np.nan))
            slack = qloc - dmin
            local_rows.append([p, qloc, dmin, slack])
        local_chk = pd.DataFrame(local_rows, columns=["product", "Local_delivered", "Local_dmin", "slack(q-dmin)"])
        local_min_slack = float(np.min(local_chk["slack(q-dmin)"].values)) if len(local_chk) else 0.0
    else:
        local_chk = None
        local_min_slack = None

    # --- Print summary ---
    print("\n=== CONSISTENCY REPORT ===")
    print(f"Tolerance: {tol:g}")

    print("\n[1] Refinery allocation: Q[p] == sum_m q_ref[p,m]")
    print(f"Max |diff|: {alloc_max:,.6g}")
    bad_alloc = alloc_chk[np.abs(alloc_chk["diff(sum_q_ref - Q)"]) > tol]
    if len(bad_alloc):
        print("Violations:")
        print(bad_alloc.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    else:
        print("OK (no violations)")

    print("\n[2] Delivered identity: q[p,m] == q_ref[p,m] + imp[p,m]")
    print(f"Max |diff|: {deliv_max:,.6g}")
    bad_deliv = deliv_chk[np.abs(deliv_chk["diff(delivered-(refinery+import))"]) > tol]
    if len(bad_deliv):
        print("Violations (showing up to 20):")
        print(bad_deliv.head(20).to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    else:
        print("OK (no violations)")

    if local_chk is not None:
        print("\n[3] Local hard mins: Local_delivered >= Local_dmin")
        print(f"Min slack (q-dmin): {local_min_slack:,.6g}")
        bad_local = local_chk[local_chk["slack(q-dmin)"] < -tol]
        if len(bad_local):
            print("Violations:")
            print(bad_local.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
        else:
            print("OK (no violations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--tol", type=float, default=1e-6, help="Consistency tolerance")
    args = ap.parse_args()

    data = load_data(args.data_dir)
    results = build_and_solve_lp(data, verbose=True)

    # Print consistency checks (does not affect solution)
    _consistency_report(data, results, tol=args.tol)


if __name__ == "__main__":
    main()
