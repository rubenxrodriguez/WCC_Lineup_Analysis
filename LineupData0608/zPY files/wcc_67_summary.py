#!/usr/bin/env python3
"""
wcc_67_summary.py

Summarize games from wcc_gamelog.csv:
- count of team-games with exactly 67 points
- count of team-games with more than 67 points
- win % when scoring exactly 67
- win % when scoring at least 67
"""

from __future__ import annotations

import pandas as pd


def _team_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["Home/Neutral", "PTS.1", "PTS", "Visitor/Neutral"]].copy()
    home.columns = ["team", "pts_for", "pts_against", "opponent"]

    away = df[["Visitor/Neutral", "PTS", "PTS.1", "Home/Neutral"]].copy()
    away.columns = ["team", "pts_for", "pts_against", "opponent"]

    out = pd.concat([home, away], ignore_index=True)
    return out


def summarize(path: str = "wcc_67/wcc_gamelog.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    team_games = _team_game_rows(df)

    team_games["win"] = team_games["pts_for"] > team_games["pts_against"]

    exactly_67 = team_games[team_games["pts_for"] == 67]
    at_least_67 = team_games[team_games["pts_for"] >= 67]
    more_than_67 = team_games[team_games["pts_for"] > 67]

    exactly_67_wins = exactly_67["win"].sum()
    at_least_67_wins = at_least_67["win"].sum()

    summary = pd.DataFrame(
        [
            {
                "metric": "count_exactly_67",
                "value": int(exactly_67.shape[0]),
            },
            {
                "metric": "count_more_than_67",
                "value": int(more_than_67.shape[0]),
            },
            {
                "metric": "win_pct_exactly_67",
                "value": float(exactly_67_wins / exactly_67.shape[0]) if exactly_67.shape[0] else 0.0,
            },
            {
                "metric": "win_pct_at_least_67",
                "value": float(at_least_67_wins / at_least_67.shape[0]) if at_least_67.shape[0] else 0.0,
            },
        ]
    )
    summary = summary.round({"value":3})
    return summary


if __name__ == "__main__":
    summary_df = summarize()
    print(summary_df.to_string(index=False))
