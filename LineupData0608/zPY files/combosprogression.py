#!/usr/bin/env python3
"""
combosprogression.py

Build combos_progression.csv from RAW stint-level lineup data (e.g. 241213.csv).
Uses the same combo logic as updated_combos.py and the lineup math from
updated_lineups.py.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---- import your existing lineup logic ----
try:
    import updated_lineups as ul  # must be in same folder or python path
except Exception as e:
    raise ImportError(
        "Could not import updated_lineups.py. Put combosprogression.py in the same folder "
        "as updated_lineups.py (or add it to PYTHONPATH). Original error:\n"
        f"{e}"
    )

try:
    import updated_combos as uc  # for combo ordering + player info
except Exception as e:
    raise ImportError(
        "Could not import updated_combos.py. Put combosprogression.py in the same folder "
        "as updated_combos.py (or add it to PYTHONPATH). Original error:\n"
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


def lineup_summary_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Interval-level lineup summary from RAW stint-level rows.
    Mirrors the structure of updated_lineups.process_lineups aggregation.
    """
    _require_cols(raw, ["secs", "ptsScored", "ptsAgst", "netPts", "oPoss", "dPoss"], "raw data")

    raw = raw.copy()
    raw["lineup"] = raw.apply(ul.create_height_sorted_lineup, axis=1)

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

    numeric_cols = agg.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    agg[numeric_cols] = agg[numeric_cols].round(3)
    agg = agg.sort_values(by="minutes", ascending=False).reset_index(drop=True)

    return agg


def _build_combo_stats_from_lineup_summary(
    lineup_df: pd.DataFrame,
    combo_size: int,
    min_minutes: Optional[float],
) -> pd.DataFrame:
    if combo_size < 2 or combo_size > 5:
        raise ValueError("combo_size must be between 2 and 5 (lineups have 5 players).")

    rows = []
    for _, row in lineup_df.iterrows():
        players = row["lineup"].split("-")
        for combo in combinations(players, combo_size):
            combo = uc.sort_players_by_height(combo, uc.PLAYER_INFO)
            out = {
                "minutes": row["minutes"],
                "pts_for": row["pts_for"],
                "pts_against": row["pts_against"],
                "o_poss": row["o_poss"],
                "d_poss": row["d_poss"],
            }
            for i, p in enumerate(combo, start=1):
                out[f"player{i}"] = p
            rows.append(out)

    combo_df = pd.DataFrame(rows)
    if combo_df.empty:
        return combo_df

    group_cols = [f"player{i}" for i in range(1, combo_size + 1)]
    combo_stats = (
        combo_df.groupby(group_cols)
        .agg(
            minutes=("minutes", "sum"),
            pts_for=("pts_for", "sum"),
            pts_against=("pts_against", "sum"),
            o_poss=("o_poss", "sum"),
            d_poss=("d_poss", "sum"),
        )
        .reset_index()
    )

    combo_stats["plus_minus"] = combo_stats["pts_for"] - combo_stats["pts_against"]
    combo_stats["off_rtg"] = (
        combo_stats["pts_for"] / combo_stats["o_poss"].replace(0, pd.NA) * 100
    ).fillna(0)
    combo_stats["def_rtg"] = (
        combo_stats["pts_against"] / combo_stats["d_poss"].replace(0, pd.NA) * 100
    ).fillna(0)
    combo_stats["net_rtg"] = combo_stats["off_rtg"] - combo_stats["def_rtg"]
    combo_stats["PM_p40"] = (
        combo_stats["plus_minus"] * 40 / combo_stats["minutes"].replace(0, pd.NA)
    ).fillna(0)

    team_plus_minus = combo_stats["plus_minus"].sum()
    team_minutes = combo_stats["minutes"].sum()
    team_o_poss = combo_stats["o_poss"].sum()
    team_d_poss = combo_stats["d_poss"].sum()
    team_pts_for = combo_stats["pts_for"].sum()
    team_pts_against = combo_stats["pts_against"].sum()

    team_off_rtg = _safe_div(team_pts_for, team_o_poss) * 100
    team_def_rtg = _safe_div(team_pts_against, team_d_poss) * 100
    team_net_rtg = team_off_rtg - team_def_rtg
    team_PM_p40 = _safe_div(team_plus_minus * 40, team_minutes)

    combo_stats["rel_PM_p40"] = combo_stats["PM_p40"] - team_PM_p40
    combo_stats["rel_off_rtg"] = combo_stats["off_rtg"] - team_off_rtg
    combo_stats["rel_def_rtg"] = combo_stats["def_rtg"] - team_def_rtg
    combo_stats["rel_net_rtg"] = combo_stats["net_rtg"] - team_net_rtg

    combo_stats["team_minutes"] = team_minutes
    combo_stats["team_plus_minus"] = team_plus_minus
    combo_stats["team_pts_for"] = team_pts_for
    combo_stats["team_pts_against"] = team_pts_against
    combo_stats["team_o_poss"] = team_o_poss
    combo_stats["team_d_poss"] = team_d_poss
    combo_stats["team_poss_total"] = team_o_poss + team_d_poss
    combo_stats["team_off_rtg"] = team_off_rtg
    combo_stats["team_def_rtg"] = team_def_rtg
    combo_stats["team_net_rtg"] = team_net_rtg
    combo_stats["team_PM_p40"] = team_PM_p40

    if min_minutes is not None:
        combo_stats = combo_stats[combo_stats["minutes"] >= min_minutes].copy()

    combo_stats = combo_stats.sort_values(["minutes", "net_rtg"], ascending=[False, False])
    return combo_stats


def build_combos_progression_csv(
    input_dir: str,
    pattern: str,
    intervals_str: str = "4,3,3,3",
    output_path: str = "Lineup Data/combos_progression.csv",
    team_id: Optional[int] = None,
    combo_sizes: Sequence[int] = (2, 3),
    min_minutes: Optional[float] = 100,
) -> pd.DataFrame:
    """
    Build combos_progression.csv from raw game recap files.

    Produces interval-level combo stats (pairs/trios by default) using the same
    combo logic as updated_combos.py. The min_minutes filter is applied per interval.
    """
    intervals = _parse_intervals(intervals_str)
    files = _discover_files(input_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files found in '{input_dir}' matching '{pattern}'.")

    chunks = _chunk_files(files, intervals)
    out_frames: List[pd.DataFrame] = []
    max_combo = max(combo_sizes) if combo_sizes else 0

    for interval_num, group in enumerate(chunks, start=1):
        dfs: List[pd.DataFrame] = []
        for fi in group:
            df = pd.read_csv(fi.path)
            df["game_yymmdd"] = fi.yymmdd
            dfs.append(df)

        raw = pd.concat(dfs, ignore_index=True)
        if team_id is not None and "teamId" in raw.columns:
            raw = raw[raw["teamId"] == team_id].copy()

        interval_summary = lineup_summary_from_raw(raw)
        games = [fi.yymmdd for fi in group]

        for combo_size in combo_sizes:
            combo_stats = _build_combo_stats_from_lineup_summary(
                interval_summary, combo_size=combo_size, min_minutes=min_minutes
            )
            if combo_stats.empty:
                continue

            combo_stats["interval_num"] = interval_num
            combo_stats["interval_start"] = games[0]
            combo_stats["interval_end"] = games[-1]
            combo_stats["interval_len"] = len(group)
            combo_stats["combo_size"] = combo_size

            for i in range(combo_size + 1, max_combo + 1):
                combo_stats[f"player{i}"] = pd.NA

            out_frames.append(combo_stats)

    if not out_frames:
        raise ValueError("No combo data produced. Check inputs or min_minutes.")

    combos_progression = pd.concat(out_frames, ignore_index=True)

    combos_progression["combo_string"] = (
        combos_progression["player1"].astype(str)
        + "-"
        + combos_progression["player2"].astype(str)
        + "-"
        + combos_progression["player3"].astype(str)
    )

    ordered_cols = [
        "combo_string",
        "interval_num",
        "minutes",
        "PM_p40",
        "rel_PM_p40",
        "rel_off_rtg",
        "rel_def_rtg",
        "rel_net_rtg",
        "pts_for",
        "pts_against",
        "o_poss",
        "d_poss",
        "plus_minus",
        "off_rtg",
        "def_rtg",
        "net_rtg",
        "team_minutes",
        "team_plus_minus",
        "team_pts_for",
        "team_pts_against",
        "team_o_poss",
        "team_d_poss",
        "team_poss_total",
        "team_off_rtg",
        "team_def_rtg",
        "team_net_rtg",
        "team_PM_p40",
        "combo_size",
        "interval_len",
        "interval_start",
        "interval_end",
        "player1",
        "player2",
        "player3",
    ]
    existing_cols = [c for c in ordered_cols if c in combos_progression.columns]
    combos_progression = combos_progression[existing_cols]
    combos_progression = combos_progression.round(3)
    combos_progression = combos_progression.drop_duplicates(['combo_string', 'interval_num'])

    combos_progression.to_csv(output_path, index=False)
    return combos_progression


if __name__ == "__main__":
    build_combos_progression_csv(
        input_dir="Game Recaps",
        pattern="*.csv",
        intervals_str="4,3,3,3,2",
        output_path="Lineup Data/combos_progression_0208.csv",
        team_id=None,
        combo_sizes=(3,3),
        min_minutes=15,
    )
    print("Wrote combos_progression.csv")
