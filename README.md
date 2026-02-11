
# AI-Assisted Risk-Aware Refinery Planning Optimization (MVP Phase 1)

This repository contains a **PIMS-like refinery planning LP** implemented in Python, with:
- Crude slate selection (multiple crudes)
- CDU yields to cuts
- Conversion units (FCC + Visbreaker/VR)
- Blending pools with **linear quality constraints** (Gasoline RON/RVP; Diesel sulfur/cetane)
- Two-market sales (Local vs Export) using **netbacks** and market demand bounds
- Deterministic single-period base case (Phase 1)

Phase 2 will add forecast-driven scenarios + CVaR risk terms.

## Quickstart

```bash
python -m pip install -r requirements.txt
python src/solve.py --data_dir data
```

## Data files

- `data/crude_yields.csv`: crude-to-cut yields for CDU
- `data/unit_yields.csv`: conversion yields for FCC and VR
- `data/qualities.csv`: blending properties for components
- `data/markets.csv`: netbacks + demand bounds by product and market
- `data/economics.csv`: crude costs, unit variable costs, unit capacities

## Model outputs (planning run report)

- crude slate (kbpd, % share)
- unit throughputs and utilizations
- product sales split by market
- blend recipes for gasoline and diesel
- quality achieved vs spec
- profit breakdown
