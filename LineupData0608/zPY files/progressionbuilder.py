#!/usr/bin/env python3
"""
progression_builder.py

Build progression.csv from RAW stint-level lineup data (e.g. 241213.csv).

Uses your existing lineup construction logic from updated_lineups.py:
- create_height_sorted_lineup (PLAYER_INFO + heights/initials)
and then applies the lineup aggregation math (same structure as updated_lineups.py)
to each interval of games.

Call:
    from progression_builder import build_progression_csv
    build_progression_csv("Game Recaps", "24*.csv", "3,2,3", output_path="progression.csv")

Assumptions:
- Filenames contain YYMMDD as a 6-digit chunk, e.g. 241213.csv
- Raw CSVs contain columns like:
  secs, ptsScored, ptsAgst, netPts, oPoss, dPoss, fgm, fga, fgm3, fga3, fta, tov, orb,
  fgmAgst, fgaAgst, fgm3Agst, fga3Agst, ftaAgst, tovAgst, orbAgst,
  plus pId1..pId5 / pName1..pName5 for lineup creation.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---- import your existing lineup logic ----
try:
    import updated_lineups as ul  # must be in same folder or python path
except Exception as e:
    raise ImportError(
        "Could not import updated_lineups.py. Put progression_builder.py in the same folder "
        "as updated_lineups.py (or add it to PYTHONPATH). Original error:\n"
        f"{e}"
    )

DATE_RE = re.compile(r"(\d{6})")  # YYMMDD


@dataclass(frozen=True)
class FileInfo:
    path: str
    yymmdd: str


def _extract_yymmdd(path: str) -> Optional[str]:
    m = DATE_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def _discover_files(input_dir: str, pattern: str) -> List[FileInfo]:
    paths = glob.glob(os.path.join(input_dir, pattern))
    infos: List[FileInfo] = []
    for p in paths:
        yymmdd = _extract_yymmdd(p)
        if yymmdd:
            infos.append(FileInfo(path=p, yymmdd=yymmdd))
    infos.sort(key=lambda x: x.yymmdd)
    return infos


def _parse_intervals(intervals_str: str) -> List[int]:
    parts = [p.strip() for p in intervals_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("intervals_str is empty. Example: '3,2,3'")
    ints = [int(p) for p in parts]
    if any(i <= 0 for i in ints):
        raise ValueError("All interval sizes must be positive integers.")
    return ints


def _chunk_files(files: Sequence[FileInfo], intervals: Sequence[int]) -> List[List[FileInfo]]:
    need = sum(intervals)
    if len(files) != need:
        raise ValueError(
            f"Expected exactly {need} files for intervals {list(intervals)}, "
            f"but found {len(files)}.\n"
            f"Matched files (sorted): {[os.path.basename(f.path) for f in files]}"
        )
    out: List[List[FileInfo]] = []
    i = 0
    for k in intervals:
        out.append(list(files[i : i + k]))
        i += k
    return out


def _safe_div(n: float, d: float) -> float:
    if d == 0 or d is None or (isinstance(d, float) and np.isnan(d)):
        return 0.0
    return float(n) / float(d)


def _require_cols(df: pd.DataFrame, cols: Sequence[str], ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns {missing} {('in ' + ctx) if ctx else ''}.\n"
            f"Columns present: {list(df.columns)}"
        )
desired = [
        "interval_num", "interval_len",
        "lineup", "minutes", "rel_PM_p40", "PM_p40", "o_poss", "d_poss",
        "plus_minus", "net_rtg", "off_rtg", "def_rtg", "poss_total",
        "rel_net_rtg",
        "team_minutes", "team_plus_minus", "team_pts_for", "team_pts_against",
        "team_o_poss", "team_d_poss", "team_poss_total", "team_off_rtg",
        "team_def_rtg", "team_net_rtg", "team_PM_p40",
        "interval_start", "interval_end",
        "pts_for", "pts_against", "net_pts",
        "o_eFG%", "d_eFG%", "o_TOV%", "d_TOV%", "o_orbR", "d_orbR", "o_ftaR", "d_ftaR",
        "rel_off_rtg", "rel_def_rtg",
    ]

def lineup_summary_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Interval-level lineup summary from RAW stint-level rows.
    Mirrors the structure of updated_lineups.process_lineups aggregation.
    """
    _require_cols(raw, ["secs", "ptsScored", "ptsAgst", "netPts", "oPoss", "dPoss"], "raw data")

    # Create lineup key using your existing logic
    raw = raw.copy()
    raw["lineup"] = raw.apply(ul.create_height_sorted_lineup, axis=1)

    # Aggregate base totals
    agg = (
        raw.groupby("lineup", dropna=False)
        .agg(
            secs=("secs", "sum"),
            pts_for=("ptsScored", "sum"),
            pts_against=("ptsAgst", "sum"),
            net_pts=("netPts", "sum"),
            o_poss=("oPoss", "sum"),
            d_poss=("dPoss", "sum"),
            fgm=("fgm", "sum"),
            fga=("fga", "sum"),
            fgm3=("fgm3", "sum"),
            fga3=("fga3", "sum"),
            fta=("fta", "sum"),
            tov=("tov", "sum"),
            orb=("orb", "sum"),
            fgm_allowed=("fgmAgst", "sum"),
            fga_allowed=("fgaAgst", "sum"),
            fgm3_allowed=("fgm3Agst", "sum"),
            fga3_allowed=("fga3Agst", "sum"),
            fta_allowed=("ftaAgst", "sum"),
            tov_forced=("tovAgst", "sum"),
            orb_allowed=("orbAgst", "sum"),
        )
        .reset_index()
    )

    # Derived
    agg["minutes"] = agg["secs"] / 60.0
    agg["poss_total"] = agg["o_poss"] + agg["d_poss"]
    agg["plus_minus"] = agg["net_pts"]

    agg["o_eFG%"] = ((agg["fgm"] + 0.5 * agg["fgm3"]) / agg["fga"].replace(0, np.nan)).fillna(0)
    agg["d_eFG%"] = (
        (agg["fgm_allowed"] + 0.5 * agg["fgm3_allowed"]) / agg["fga_allowed"].replace(0, np.nan)
    ).fillna(0)

    agg["o_TOV%"] = (agg["tov"] / agg["o_poss"].replace(0, np.nan)).fillna(0)
    agg["d_TOV%"] = (agg["tov_forced"] / agg["d_poss"].replace(0, np.nan)).fillna(0)

    agg["o_orbR"] = (agg["orb"] / agg["o_poss"].replace(0, np.nan)).fillna(0)
    agg["d_orbR"] = (agg["orb_allowed"] / agg["d_poss"].replace(0, np.nan)).fillna(0)

    agg["o_ftaR"] = (agg["fta"] / agg["fga"].replace(0, np.nan)).fillna(0)
    agg["d_ftaR"] = (agg["fta_allowed"] / agg["fga_allowed"].replace(0, np.nan)).fillna(0)

    agg["off_rtg"] = (agg["pts_for"] / agg["o_poss"].replace(0, np.nan) * 100).fillna(0)
    agg["def_rtg"] = (agg["pts_against"] / agg["d_poss"].replace(0, np.nan) * 100).fillna(0)
    agg["net_rtg"] = agg["off_rtg"] - agg["def_rtg"]

    agg["PM_p40"] = (
        agg["plus_minus"] * 40 / agg["minutes"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Team totals (interval context)
    team_plus_minus = float(agg["plus_minus"].sum())
    team_minutes = float(agg["minutes"].sum())
    team_o_poss = float(agg["o_poss"].sum())
    team_d_poss = float(agg["d_poss"].sum())
    team_pts_for = float(agg["pts_for"].sum())
    team_pts_against = float(agg["pts_against"].sum())

    team_off_rtg = _safe_div(team_pts_for, team_o_poss) * 100
    team_def_rtg = _safe_div(team_pts_against, team_d_poss) * 100
    team_net_rtg = team_off_rtg - team_def_rtg
    team_PM_p40 = _safe_div(team_plus_minus * 40, team_minutes)

    agg["rel_PM_p40"] = agg["PM_p40"] - team_PM_p40
    agg["rel_off_rtg"] = agg["off_rtg"] - team_off_rtg
    agg["rel_def_rtg"] = agg["def_rtg"] - team_def_rtg
    agg["rel_net_rtg"] = agg["net_rtg"] - team_net_rtg

    agg["team_minutes"] = team_minutes
    agg["team_plus_minus"] = team_plus_minus
    agg["team_pts_for"] = team_pts_for
    agg["team_pts_against"] = team_pts_against
    agg["team_o_poss"] = team_o_poss
    agg["team_d_poss"] = team_d_poss
    agg["team_poss_total"] = team_o_poss + team_d_poss
    agg["team_off_rtg"] = team_off_rtg
    agg["team_def_rtg"] = team_def_rtg
    agg["team_net_rtg"] = team_net_rtg
    agg["team_PM_p40"] = team_PM_p40

    # Round + sort
    numeric_cols = agg.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    agg[numeric_cols] = agg[numeric_cols].round(3)
    agg = agg.sort_values(by="minutes", ascending=False).reset_index(drop=True)

    return agg


def build_progression_csv(
    input_dir: str,
    pattern: str,
    intervals_str: str = "3,2,3",
    output_path: str = "progression.csv",
    team_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build progression.csv from raw game recap files.

    team_id:
      If your raw files contain multiple teams, pass the LMU teamId.
      If None, no filter is applied.
    """
    intervals = _parse_intervals(intervals_str)
    files = _discover_files(input_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files found in '{input_dir}' matching '{pattern}'.")

    chunks = _chunk_files(files, intervals)
    out_frames: List[pd.DataFrame] = []

    for interval_num, group in enumerate(chunks, start=1):
        dfs: List[pd.DataFrame] = []
        for fi in group:
            df = pd.read_csv(fi.path)
            df["game_yymmdd"] = fi.yymmdd
            dfs.append(df)

        raw = pd.concat(dfs, ignore_index=True)

        if team_id is not None and "teamId" in raw.columns:
            raw = raw[raw["teamId"] == team_id].copy()

        # Build interval lineup summary with true underlying math
        interval_summary = lineup_summary_from_raw(raw)

        # Add interval metadata
        games = [fi.yymmdd for fi in group]
        interval_summary["interval_num"] = interval_num
        interval_summary["interval_start"] = games[0]
        interval_summary["interval_end"] = games[-1]
        interval_summary["interval_len"] = len(group)  

        interval_summary = interval_summary[desired]

        out_frames.append(interval_summary)

    progression = pd.concat(out_frames, ignore_index=True)
    progression.to_csv(output_path, index=False)
    return progression


# Optional: run directly if you want, but not required
if __name__ == "__main__":
    build_progression_csv(
        input_dir="Game Recaps",
        pattern="*.csv",
        intervals_str="4,3,3,3,2",
        output_path="Lineup Data/progression0208.csv",
        team_id=None,
    )
    print("Wrote progression0207.csv")