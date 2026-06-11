#!/usr/bin/env python3
"""
lineup_meta.py

Export lineup metadata per game from raw recap CSVs:
- lineup_count
- average/min/max/median/stdev minutes per lineup

Default output: Lineup Data/lineup_meta.csv
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import updated_lineups as ul


def _lineup_minutes(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["lineup"] = df.apply(ul.create_height_sorted_lineup, axis=1)
    return df.groupby("lineup")["secs"].sum() / 60.0


def build_lineup_meta(
    input_dir: str = "Game Recaps",
    pattern: str = "*.csv",
    output_path: str = "Lineup Data/lineup_meta.csv",
) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files found in '{input_dir}' matching '{pattern}'.")

    rows: List[dict] = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        lineup_minutes = _lineup_minutes(df)
        if lineup_minutes.empty:
            continue

        rows.append(
            {
                "file": os.path.basename(p),
                "lineup_count": int(lineup_minutes.shape[0]),
                "non_unique_lineup_count": int(df.shape[0]),
                "minutes_mean": float(lineup_minutes.mean()),
                "minutes_median": float(lineup_minutes.median()),
                "minutes_min": float(lineup_minutes.min()),
                "minutes_max": float(lineup_minutes.max()),
                "minutes_std": float(lineup_minutes.std(ddof=1)),
            }
        )

    
    out = pd.DataFrame(rows)
    out["lineup_density"] = out["non_unique_lineup_count"] / out["lineup_count"]
    if out.empty:
        raise ValueError("No lineup metadata produced. Check inputs.")
    num_cols = out.select_dtypes(include = "number").columns
    out[num_cols] = out[num_cols].round(2)
    out = out.sort_values("file").reset_index(drop=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    build_lineup_meta(
        input_dir="Game Recaps",
        pattern="*.csv",
        output_path="Lineup Data/lineup_meta.csv",
    )
    print("Wrote Lineup Data/lineup_meta.csv")
