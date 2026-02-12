from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Specs:
    # Gasoline
    RON_min: float = 87.0
    RVP_max: float = 9.0
    # Diesel
    S_max_ppm: float = 1000.0
    CN_min: float = 45.0


@dataclass(frozen=True)
class Data:
    crude_yields: pd.DataFrame
    unit_yields: pd.DataFrame
    qualities: pd.DataFrame
    markets: pd.DataFrame
    economics: pd.DataFrame
    imports: pd.DataFrame
    feed_imports: pd.DataFrame
    specs: Specs = Specs()


def load_data(data_dir):
    crude_yields = pd.read_csv(f"{data_dir}/crude_yields.csv")
    unit_yields = pd.read_csv(f"{data_dir}/unit_yields.csv")
    qualities = pd.read_csv(f"{data_dir}/qualities.csv")
    markets = pd.read_csv(f"{data_dir}/markets.csv")
    economics = pd.read_csv(f"{data_dir}/economics.csv")

    imports_path = f"{data_dir}/imports.csv"
    if os.path.exists(imports_path):
        imports = pd.read_csv(imports_path)
    else:
        imports = pd.DataFrame(columns=["product", "market", "import_cost", "import_max"])

    feed_imports_path = f"{data_dir}/feed_imports.csv"
    if os.path.exists(feed_imports_path):
        feed_imports = pd.read_csv(feed_imports_path)
    else:
        feed_imports = pd.DataFrame(columns=["feed", "import_cost", "import_max"])

    return Data(
        crude_yields=crude_yields,
        unit_yields=unit_yields,
        qualities=qualities,
        markets=markets,
        economics=economics,
        imports=imports,
        feed_imports=feed_imports,
    )


def _econ_lookup(econ, typ):
    sub = econ[econ["type"] == typ].copy()
    return dict(zip(sub["name"], sub["value"]))


def build_and_solve_lp(data, verbose=True):
    """
    Refinery planning LP with:
      - Local minimum demand HARD (served by delivered volumes = refinery + finished imports)
      - Export minimum demand soft (unmet allowed)
      - FCC has two severity modes (FCC_G and FCC_D)
      - HC competes for VGO with FCC
      - HDT upgrades selected diesel blendstocks into treated diesel stream D_TRT
      - Feed imports: e.g., VGO import to supplement CDU VGO production
      - Hydrogen balance (H2 availability + optional H2 purchase)

    Accounting fix:
      - q_ref[p,m] = refinery-delivered volume to market m
      - q[p,m]     = delivered volume to market m (refinery + imports)
      - Revenue is earned on delivered q[p,m]
    """

    # --- Sets ---
    crudes = list(data.crude_yields["crude"].unique())

    cuts = ["N", "K", "D", "VGO", "R"]
    conv_streams = ["G_FCC", "D_FCC", "LPG", "FO", "D_VR", "RFG", "D_HC", "N_HC", "D_TRT"]
    streams = cuts + conv_streams

    products = ["G", "J", "U", "L", "F"]
    markets = list(data.markets["market"].unique())
    local_market_name = "Local"
    export_markets = [m for m in markets if m != local_market_name]

    # HDT treatable diesel blendstocks
    treatable = ["D_FCC", "D_VR", "D_HC"]

    # Blending eligibility
    blend_map = {
        "G": ["N", "N_HC", "G_FCC", "RFG"],
        "J": ["K"],
        "U": ["D", "D_FCC", "D_VR", "D_HC", "D_TRT"],
        "L": ["LPG"],
        "F": ["FO"],
    }

    # --- Parameters ---
    crude_y = data.crude_yields.set_index("crude")[cuts].to_dict(orient="index")
    unit_y = data.unit_yields.set_index(["unit", "stream"])["yield"].to_dict()

    qtab = data.qualities.set_index("stream")
    qual = {}
    for s in qtab.index:
        qual[s] = {
            "RON": float(qtab.loc[s, "RON"]) if not pd.isna(qtab.loc[s, "RON"]) else None,
            "RVP": float(qtab.loc[s, "RVP"]) if not pd.isna(qtab.loc[s, "RVP"]) else None,
            "S_ppm": float(qtab.loc[s, "S_ppm"]) if not pd.isna(qtab.loc[s, "S_ppm"]) else None,
            "CN": float(qtab.loc[s, "CN"]) if not pd.isna(qtab.loc[s, "CN"]) else None,
        }

    netback = data.markets.set_index(["product", "market"])["netback"].to_dict()
    dmin = data.markets.set_index(["product", "market"])["demand_min"].to_dict()
    dmax = data.markets.set_index(["product", "market"])["demand_max"].to_dict()

    crude_cost = _econ_lookup(data.economics, "crude_cost")
    unit_cost = _econ_lookup(data.economics, "unit_cost")
    cap = _econ_lookup(data.economics, "capacity")

    # Hydrogen parameters
    h2_rate = _econ_lookup(data.economics, "h2_rate") if "h2_rate" in set(data.economics["type"]) else {}
    h2_supply = _econ_lookup(data.economics, "h2_supply") if "h2_supply" in set(data.economics["type"]) else {}
    h2_cost = _econ_lookup(data.economics, "h2_cost") if "h2_cost" in set(data.economics["type"]) else {}
    h2_cap = _econ_lookup(data.economics, "h2_cap") if "h2_cap" in set(data.economics["type"]) else {}

    # Defaults so model still runs if you forget rows
    h2_rate_HC = float(h2_rate.get("HC", 0.0))
    h2_rate_HDT = float(h2_rate.get("HDT", 0.0))
    h2_avail = float(h2_supply.get("avail", 0.0))
    h2_buy_cost = float(h2_cost.get("buy", 0.0))
    h2_buy_cap = float(h2_cap.get("buy", 0.0))

    specs = data.specs

    # Export unmet penalty (Local unmet not allowed)
    penalty_unmet = {"G": 500.0, "J": 400.0, "U": 600.0, "L": 200.0, "F": 50.0}

    # Finished imports (optional)
    imp_cost: Dict[Tuple[str, str], float] = {}
    imp_max: Dict[Tuple[str, str], float] = {}
    if len(data.imports) > 0:
        tmp = data.imports.copy()
        if "import_max" not in tmp.columns:
            tmp["import_max"] = 1e9
        for _, r in tmp.iterrows():
            key = (str(r["product"]), str(r["market"]))
            imp_cost[key] = float(r["import_cost"])
            imp_max[key] = float(r["import_max"])

    # Feed imports (optional) (e.g., VGO)
    feed_cost: Dict[str, float] = {}
    feed_max: Dict[str, float] = {}
    if len(data.feed_imports) > 0:
        tmp = data.feed_imports.copy()
        if "import_max" not in tmp.columns:
            tmp["import_max"] = 1e9
        for _, r in tmp.iterrows():
            feed = str(r["feed"])
            feed_cost[feed] = float(r["import_cost"])
            feed_max[feed] = float(r["import_max"])

    # --- Decision variables ---
    var_names: List[str] = []
    idx: Dict[str, int] = {}

    def add_var(name):
        idx[name] = len(var_names)
        var_names.append(name)

    # Crude runs
    for c in crudes:
        add_var(f"x[{c}]")

    # Unit throughputs
    add_var("T[FCC_G]")
    add_var("T[FCC_D]")
    add_var("T[VR]")
    add_var("T[REF]")
    add_var("T[HC]")
    add_var("T[HDT]")

    # Hydrogen purchase
    add_var("H2[buy]")

    # Feed imports (e.g., VGO)
    for feed in sorted(feed_cost.keys()):
        add_var(f"feed_imp[{feed}]")

    # Stream availabilities
    for s in streams:
        add_var(f"y[{s}]")

    # HDT treating amounts
    for s in treatable:
        add_var(f"treat[{s}]")

    # Blending flows
    for p in products:
        for s in blend_map[p]:
            add_var(f"f[{s}->{p}]")

    # Total products
    for p in products:
        add_var(f"Q[{p}]")

    # Refinery-delivered sales by market (excludes finished imports)
    for p in products:
        for m in markets:
            add_var(f"q_ref[{p},{m}]")

    # Delivered sales by market (refinery + imports)
    # (kept as q[...] so the rest of the model / reporting stays familiar)
    for p in products:
        for m in markets:
            add_var(f"q[{p},{m}]")

    # Finished imports
    for (p, m) in sorted(imp_cost.keys()):
        add_var(f"imp[{p},{m}]")

    # Export unmet only
    for p in products:
        for m in export_markets:
            add_var(f"unmet[{p},{m}]")

    n = len(var_names)

    # Bounds
    bounds: List[Tuple[float, float]] = [(0, None)] * n

    # Delivered sales bounds
    for p in products:
        for m in markets:
            bounds[idx[f"q[{p},{m}]"]] = (0.0, float(dmax[(p, m)]))

    # Finished import bounds
    for (p, m), mx in imp_max.items():
        bounds[idx[f"imp[{p},{m}]"]] = (0.0, float(mx))

    # Feed import bounds
    for feed, mx in feed_max.items():
        bounds[idx[f"feed_imp[{feed}]"]] = (0.0, float(mx))

    # H2 buy bound
    bounds[idx["H2[buy]"]] = (0.0, float(h2_buy_cap))

    # --- Objective (minimize negative profit) ---
    cvec = np.zeros(n)

    # Revenues (delivered volumes earn revenue)
    for p in products:
        for m in markets:
            cvec[idx[f"q[{p},{m}]"]] = -float(netback[(p, m)])

    # Finished import costs
    for (p, m), cost in imp_cost.items():
        cvec[idx[f"imp[{p},{m}]"]] += float(cost)

    # Feed import costs
    for feed, cost in feed_cost.items():
        cvec[idx[f"feed_imp[{feed}]"]] += float(cost)

    # Export unmet penalty
    for p in products:
        for m in export_markets:
            cvec[idx[f"unmet[{p},{m}]"]] = float(penalty_unmet[p])

    # Crude costs + CDU variable cost
    for c in crudes:
        cvec[idx[f"x[{c}]"]] = float(crude_cost[c]) + float(unit_cost["CDU"])

    # Unit variable costs
    cvec[idx["T[FCC_G]"]] += float(unit_cost["FCC"])
    cvec[idx["T[FCC_D]"]] += float(unit_cost["FCC"])
    cvec[idx["T[VR]"]] += float(unit_cost["VR"])
    cvec[idx["T[REF]"]] += float(unit_cost["REF"])
    cvec[idx["T[HC]"]] += float(unit_cost["HC"])
    cvec[idx["T[HDT]"]] += float(unit_cost["HDT"])

    # Hydrogen purchase cost
    cvec[idx["H2[buy]"]] += float(h2_buy_cost)

    # --- Constraints ---
    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    def eq(row, rhs):
        r = np.zeros(n)
        for name, val_ in row.items():
            r[idx[name]] = val_
        A_eq.append(r)
        b_eq.append(rhs)

    def ub(row, rhs):
        r = np.zeros(n)
        for name, val_ in row.items():
            r[idx[name]] = val_
        A_ub.append(r)
        b_ub.append(rhs)

    # 1) CDU capacity
    ub({f"x[{c}]": 1.0 for c in crudes}, float(cap["CDU"]))

    # 2) CDU yields
    for s in cuts:
        row = {f"y[{s}]": 1.0}
        for c in crudes:
            row[f"x[{c}]"] = -float(crude_y[c][s])
        eq(row, 0.0)

    # 3) VGO allocation: FCC + HC <= VGO + VGO_import
    vgo_row = {"T[FCC_G]": 1.0, "T[FCC_D]": 1.0, "T[HC]": 1.0, "y[VGO]": -1.0}
    if "VGO" in feed_cost:
        vgo_row["feed_imp[VGO]"] = -1.0
    ub(vgo_row, 0.0)

    # 4) VR feed <= resid
    ub({"T[VR]": 1.0, "y[R]": -1.0}, 0.0)

    # 5) Conversion yields
    eq(
        {
            "y[G_FCC]": 1.0,
            "T[FCC_G]": -float(unit_y[("FCC_G", "G_FCC")]),
            "T[FCC_D]": -float(unit_y[("FCC_D", "G_FCC")]),
        },
        0.0,
    )
    eq(
        {
            "y[D_FCC]": 1.0,
            "T[FCC_G]": -float(unit_y[("FCC_G", "D_FCC")]),
            "T[FCC_D]": -float(unit_y[("FCC_D", "D_FCC")]),
        },
        0.0,
    )
    eq(
        {
            "y[LPG]": 1.0,
            "T[FCC_G]": -float(unit_y[("FCC_G", "LPG")]),
            "T[FCC_D]": -float(unit_y[("FCC_D", "LPG")]),
        },
        0.0,
    )

    eq({"y[D_HC]": 1.0, "T[HC]": -float(unit_y[("HC", "D_HC")])}, 0.0)
    eq({"y[N_HC]": 1.0, "T[HC]": -float(unit_y[("HC", "N_HC")])}, 0.0)

    eq({"y[RFG]": 1.0, "T[REF]": -float(unit_y[("REF", "RFG")])}, 0.0)

    eq({"y[FO]": 1.0, "T[VR]": -float(unit_y[("VR", "FO")])}, 0.0)
    eq({"y[D_VR]": 1.0, "T[VR]": -float(unit_y[("VR", "D_VR")])}, 0.0)

    # 5b) HDT: D_TRT = sum treat[s]
    eq({"y[D_TRT]": 1.0, **{f"treat[{s}]": -1.0 for s in treatable}}, 0.0)

    # 5c) Define T[HDT] = sum treat[s]
    eq({"T[HDT]": 1.0, **{f"treat[{s}]": -1.0 for s in treatable}}, 0.0)

    # 6) Unit capacities
    ub({"T[FCC_G]": 1.0, "T[FCC_D]": 1.0}, float(cap["FCC"]))
    ub({"T[VR]": 1.0}, float(cap["VR"]))
    ub({"T[REF]": 1.0}, float(cap["REF"]))
    ub({"T[HC]": 1.0}, float(cap["HC"]))
    ub({"T[HDT]": 1.0}, float(cap["HDT"]))

    # 6b) Hydrogen balance: H2_req <= H2_avail + H2_buy
    ub({"T[HC]": h2_rate_HC, "T[HDT]": h2_rate_HDT, "H2[buy]": -1.0}, float(h2_avail))

    # 7) Stream availability (treating removes material; N to reformer)
    for s in streams:
        used = [(s, p) for p in products if s in blend_map[p]]
        if not used and s != "N":
            continue

        row = {f"y[{s}]": -1.0}
        for s2, p in used:
            row[f"f[{s2}->{p}]"] = 1.0

        if s in treatable:
            row[f"treat[{s}]"] = 1.0

        if s == "N":
            row["T[REF]"] = 1.0

        ub(row, 0.0)

    # 8) Product pool definition
    for p in products:
        row = {f"Q[{p}]": 1.0}
        for s in blend_map[p]:
            row[f"f[{s}->{p}]"] = -1.0
        eq(row, 0.0)

    # 9) Refinery sales balance (Q is refinery production only)
    for p in products:
        row = {f"Q[{p}]": 1.0}
        for m in markets:
            row[f"q_ref[{p},{m}]"] = -1.0
        eq(row, 0.0)

    # 9a) Delivered sales definition: delivered = refinery + finished imports
    for p in products:
        for m in markets:
            row = {f"q[{p},{m}]": 1.0, f"q_ref[{p},{m}]": -1.0}
            if (p, m) in imp_cost:
                row[f"imp[{p},{m}]"] = -1.0
            eq(row, 0.0)

    # 9b) HARD Local minimums (delivered): q >= dmin  -> -q <= -dmin
    for p in products:
        m = local_market_name
        ub({f"q[{p},{m}]": -1.0}, -float(dmin[(p, m)]))

    # 9c) SOFT Export minimums (delivered): q + unmet >= dmin
    for p in products:
        for m in export_markets:
            ub({f"q[{p},{m}]": -1.0, f"unmet[{p},{m}]": -1.0}, -float(dmin[(p, m)]))

    # 10) Blending specs
    row = {"Q[G]": float(specs.RON_min)}
    for s in blend_map["G"]:
        if qual[s]["RON"] is None:
            raise ValueError(f"Missing RON for gasoline component {s}")
        row[f"f[{s}->G]"] = -float(qual[s]["RON"])
    ub(row, 0.0)

    row = {"Q[G]": -float(specs.RVP_max)}
    for s in blend_map["G"]:
        if qual[s]["RVP"] is None:
            raise ValueError(f"Missing RVP for gasoline component {s}")
        row[f"f[{s}->G]"] = float(qual[s]["RVP"])
    ub(row, 0.0)

    row = {"Q[U]": -float(specs.S_max_ppm)}
    for s in blend_map["U"]:
        if qual[s]["S_ppm"] is None:
            raise ValueError(f"Missing sulfur for diesel component {s}")
        row[f"f[{s}->U]"] = float(qual[s]["S_ppm"])
    ub(row, 0.0)

    row = {"Q[U]": float(specs.CN_min)}
    for s in blend_map["U"]:
        if qual[s]["CN"] is None:
            raise ValueError(f"Missing CN for diesel component {s}")
        row[f"f[{s}->U]"] = -float(qual[s]["CN"])
    ub(row, 0.0)

    # --- Solve ---
    res = linprog(
        c=cvec,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP solve failed: {res.message}")

    x = res.x

    def val(name: str) -> float:
        return float(x[idx[name]])

    # --- Reporting ---
    crude_df = pd.DataFrame({"crude": crudes, "run_kbpd": [val(f"x[{c}]") for c in crudes]})
    crude_df["share_%"] = 100 * crude_df["run_kbpd"] / max(crude_df["run_kbpd"].sum(), 1e-9)

    fcc_g = val("T[FCC_G]")
    fcc_d = val("T[FCC_D]")
    fcc_total = fcc_g + fcc_d
    fcc_split_df = pd.DataFrame(
        [["FCC_G", fcc_g, 100 * fcc_g / max(fcc_total, 1e-9)], ["FCC_D", fcc_d, 100 * fcc_d / max(fcc_total, 1e-9)]],
        columns=["mode", "throughput_kbpd", "share_%"],
    )

    T_cdu = crude_df["run_kbpd"].sum()
    unit_df = pd.DataFrame(
        [
            ["CDU", T_cdu, cap["CDU"]],
            ["FCC", fcc_total, cap["FCC"]],
            ["VR", val("T[VR]"), cap["VR"]],
            ["REF", val("T[REF]"), cap["REF"]],
            ["HC", val("T[HC]"), cap["HC"]],
            ["HDT", val("T[HDT]"), cap["HDT"]],
        ],
        columns=["unit", "throughput_kbpd", "capacity_kbpd"],
    )
    unit_df["util_%"] = 100 * unit_df["throughput_kbpd"] / unit_df["capacity_kbpd"]

    # Product table
    prod_rows = []
    for p in products:
        row = {"product": p, "refinery_kbpd": val(f"Q[{p}]")}

        m = local_market_name
        row[f"{m}_delivered_kbpd"] = val(f"q[{p},{m}]")
        row[f"{m}_refinery_kbpd"] = val(f"q_ref[{p},{m}]")
        row[f"{m}_import_kbpd"] = val(f"imp[{p},{m}]") if (p, m) in imp_cost else 0.0
        row[f"{m}_unmet_kbpd"] = 0.0

        for em in export_markets:
            row[f"{em}_delivered_kbpd"] = val(f"q[{p},{em}]")
            row[f"{em}_refinery_kbpd"] = val(f"q_ref[{p},{em}]")
            row[f"{em}_import_kbpd"] = val(f"imp[{p},{em}]") if (p, em) in imp_cost else 0.0
            row[f"{em}_unmet_kbpd"] = val(f"unmet[{p},{em}]")

        prod_rows.append(row)
    prod_df = pd.DataFrame(prod_rows)

    # Feed imports report
    vgo_imp = val("feed_imp[VGO]") if "feed_imp[VGO]" in idx else 0.0

    # Hydrogen report
    h2_req = h2_rate_HC * val("T[HC]") + h2_rate_HDT * val("T[HDT]")
    h2_buy = val("H2[buy]")
    h2_df = pd.DataFrame(
        [[h2_req, h2_avail, h2_buy, max(h2_req - (h2_avail + h2_buy), 0.0)]],
        columns=["H2_required", "H2_free_avail", "H2_purchased", "H2_short"],
    )

    # Economics (revenue on delivered volumes)
    revenue = sum(netback[(p, m)] * val(f"q[{p},{m}]") for p in products for m in markets)
    crude_cost_total = sum(crude_cost[c] * val(f"x[{c}]") for c in crudes)

    opex = (
        unit_cost["CDU"] * T_cdu
        + unit_cost["FCC"] * fcc_total
        + unit_cost["VR"] * val("T[VR]")
        + unit_cost["REF"] * val("T[REF]")
        + unit_cost["HC"] * val("T[HC]")
        + unit_cost["HDT"] * val("T[HDT]")
    )
    finished_import_cost_total = sum(imp_cost[(p, m)] * val(f"imp[{p},{m}]") for (p, m) in imp_cost)
    feed_import_cost_total = sum(feed_cost[feed] * val(f"feed_imp[{feed}]") for feed in feed_cost)
    h2_purchase_cost_total = h2_buy_cost * h2_buy
    unmet_pen = sum(penalty_unmet[p] * val(f"unmet[{p},{m}]") for p in products for m in export_markets)

    profit = (
        revenue
        - crude_cost_total
        - opex
        - finished_import_cost_total
        - feed_import_cost_total
        - h2_purchase_cost_total
        - unmet_pen
    )

    econ_df = pd.DataFrame(
        [
            ["Revenue (delivered)", revenue],
            ["Crude cost", -crude_cost_total],
            ["Variable OPEX", -opex],
            ["Finished import cost", -finished_import_cost_total],
            ["Feed import cost", -feed_import_cost_total],
            ["Hydrogen purchase cost", -h2_purchase_cost_total],
            ["Unmet export penalty", -unmet_pen],
            ["Profit", profit],
        ],
        columns=["item", "USDk_per_day"],
    )

    if verbose:
        print("\n=== CRUDE SLATE ===")
        print(crude_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        print("\n=== UNIT THROUGHPUTS ===")
        print(unit_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        print("\n=== FCC MODE SPLIT ===")
        print(fcc_split_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        if "VGO" in feed_cost:
            print("\n=== FEED IMPORTS ===")
            print(f"VGO import (kbpd): {vgo_imp:,.2f}")

        print("\n=== HYDROGEN BALANCE ===")
        print(h2_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        print("\n=== PRODUCT DELIVERIES (split refinery vs imports) ===")
        print(prod_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        print("\n=== ECONOMICS ===")
        print(econ_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

        print("\n(Solved with HiGHS via scipy.optimize.linprog)")

    return {
        "crude": crude_df,
        "fcc_split": fcc_split_df,
        "units": unit_df,
        "hydrogen": h2_df,
        "products": prod_df,
        "economics": econ_df,
        "raw": {"status": res.status, "message": res.message, "objective_min": float(res.fun)},
    }
