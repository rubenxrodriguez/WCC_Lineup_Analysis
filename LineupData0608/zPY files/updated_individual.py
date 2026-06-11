import numpy as np
import pandas as pd


def summarize_individuals(
    lineup_summary_path="Lineup Data/lineup_summary_inconf12.csv",
    output_path="Lineup Data/individual_summary_inconf12.csv",
):
    """
    Build individual PM and efficiency splits from a lineup summary produced by updated_lineups_copy.py.
    """
    df = pd.read_csv(lineup_summary_path)
    if df.empty:
        print("Lineup summary is empty; no individual stats computed.")
        return None

    # Split and explode to one row per player per lineup
    exploded = df.assign(players=df["lineup"].str.split("-")).explode("players")

    player_summary = (
        exploded.groupby("players").agg(
            plus_minus=("plus_minus", "sum"),
            o_poss=("o_poss", "sum"),
            d_poss=("d_poss", "sum"),
            pts_for=("pts_for", "sum"),
            pts_against=("pts_against", "sum"),
            minutes=("minutes", "sum"),
        )
    ).reset_index()

    player_summary["off_rtg"] = np.where(
        player_summary["o_poss"] > 0,
        (player_summary["pts_for"] / player_summary["o_poss"]) * 100,
        np.nan,
    )
    player_summary["def_rtg"] = np.where(
        player_summary["d_poss"] > 0,
        (player_summary["pts_against"] / player_summary["d_poss"]) * 100,
        np.nan,
    )
    player_summary["net_rtg"] = player_summary["off_rtg"] - player_summary["def_rtg"]
    player_summary["PM_p40"] = player_summary["plus_minus"] / player_summary["minutes"] * 40

    # team benchmarks copied from lineup summary (same on every row)
    player_summary["team_PM_p40"] = df["team_PM_p40"].iloc[0]
    player_summary["team_off_rtg"] = df["team_off_rtg"].iloc[0]
    player_summary["team_def_rtg"] = df["team_def_rtg"].iloc[0]

    # Round for readability
    player_summary = player_summary.round(
        {
            "off_rtg": 2,
            "def_rtg": 2,
            "o_poss": 1,
            "d_poss": 1,
            "plus_minus": 1,
            "minutes": 1,
            "net_rtg": 2,
            "PM_p40": 2,
            "team_PM_p40": 2,
            "team_off_rtg": 2,
            "team_def_rtg": 2,
        }
    )

    player_summary = player_summary.sort_values("PM_p40", ascending=False)
    player_summary.to_csv(output_path, index=False)
    print(f"Exported individual summary to {output_path}")
    return player_summary


if __name__ == "__main__":
    summary = summarize_individuals()
    if summary is not None:
        print(summary.head())

