import numpy as np
import pandas as pd
from pathlib import Path

# --- Player Info Dictionary (heights in inches) ---
# Keyed by official PID so we can join lineup rows that reference pId1..pId5.
PLAYER_INFO = {
    "2138783": {"pid": "2138783", "name": "Jess Lawson", "initial": "JL", "height": 67},
    "2118049": {"pid": "2118049", "name": "Mari Somvichian", "initial": "MS", "height": 64},
    "1925105": {"pid": "1925105", "name": "Andjela Matic", "initial": "AM", "height": 69},
    "2270840": {"pid": "2270840", "name": "Ivana Krajina", "initial": "IK", "height": 71},
    "2291264": {"pid": "2291264", "name": "Zawadi Ogot", "initial": "ZO", "height": 71},
    "2283482": {"pid": "2283482", "name": "Lova Lagerlid", "initial": "LL", "height": 73},
    "1688590": {"pid": "1688590", "name": "Carly Heidger", "initial": "CH", "height": 75},
    "2481512": {"pid": "2481512", "name": "Kayla Jones", "initial": "KJ", "height": 72},
    "2118064": {"pid": "2118064", "name": "Maya Hernandez", "initial": "MH", "height": 72},
    "2283300": {"pid": "2283300", "name": "Ana Milanovic", "initial": "AM7", "height": 75},
    "1696172": {"pid": "1696172", "name": "Paula Reus Piza", "initial": "PR", "height": 74},
    "2291259": {"pid": "2291259", "name": "Allison Clarke", "initial": "AC", "height": 71},
}


def _safe_height(val):
    return val if pd.notnull(val) else float("inf")


def _fallback_initial(name, pid):
    """Compact label if a player isn't in PLAYER_INFO."""
    if isinstance(name, str) and name.strip():
        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][:1] + parts[1][:2]).upper()
    return str(pid)


def create_height_sorted_lineup(row):
    """Create a consistent lineup string ordered by player height."""
    players = []
    for i in range(1, 6):
        pid_key = f"pId{i}"
        name_key = f"pName{i}"

        pid = row.get(pid_key)
        pid = "" if pd.isna(pid) else str(pid).strip()

        name = row.get(name_key, "")
        info = PLAYER_INFO.get(pid)

        initial = info["initial"] if info else _fallback_initial(name, pid)
        height = _safe_height(info["height"] if info else np.nan)
        players.append({"initial": initial, "height": height})

    sorted_players = sorted(players, key=lambda p: (p["height"], p["initial"]))
    return "-".join(p["initial"] for p in sorted_players)


def _safe_div(numer, denom):
    return numer / denom if denom else 0


def load_game_recaps(base_dir="Game Recaps", games=None):
    """
    Load and concatenate lineup flow CSVs from a directory, adding a `game` column.
    games: optional list of game names (matching file stems, case-insensitive) to include.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    game_filter = set(g.lower() for g in games) if games else None
    frames = []

    for csv_path in sorted(base_path.glob("*.csv")):
        game_name = csv_path.stem.lower()
        if game_filter and game_name not in game_filter:
            continue
        import pandas as pd

        pid_cols = [f"pId{i}" for i in range(1, 6)]

        df = pd.read_csv(
            csv_path,
            dtype={c: "string" for c in pid_cols}  # <- critical
)

        df["game"] = game_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def process_lineups(base_dir="Game Recaps", games=None, team_id=None):
    """
    Process multiple game recap CSVs and return aggregated lineup metrics.

    base_dir: folder containing game recap CSVs.
    games: optional list of game names to include (match CSV stems, e.g., ["utahstate","wichita"]).
    team_id: optional numeric filter if files contain multiple teams.
    """
    df = load_game_recaps(base_dir=base_dir, games=games)
    if df.empty:
        print("No lineup stints found for the given filters.")
        return None

    if team_id is not None:
        df = df[df["teamId"] == team_id]

    if df.empty:
        print("No lineup stints remain after filtering by team.")
        return None

    df["lineup"] = df.apply(create_height_sorted_lineup, axis=1)

    agg = (
        df.groupby("lineup")
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

    agg["minutes"] = agg["secs"] / 60
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

    team_plus_minus = agg["plus_minus"].sum()
    team_minutes = agg["minutes"].sum()
    team_o_poss = agg["o_poss"].sum()
    team_d_poss = agg["d_poss"].sum()
    team_pts_for = agg["pts_for"].sum()
    team_pts_against = agg["pts_against"].sum()

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

    numeric_cols = agg.select_dtypes(include=["float64", "int64"]).columns
    agg[numeric_cols] = agg[numeric_cols].round(3)
    agg = agg.sort_values(by="minutes", ascending=False).reset_index(drop=True)
    return agg[
        [
            "lineup",
            "poss_total",
            "PM_p40",
            "plus_minus",
            "net_rtg",
            "minutes",
            "o_poss",
            "d_poss",
            "pts_for",
            "pts_against",
            "o_eFG%",
            "o_TOV%",
            "o_orbR",
            "o_ftaR",
            "off_rtg",
            "d_eFG%",
            "d_TOV%",
            "d_orbR",
            "d_ftaR",
            "def_rtg",
            "rel_PM_p40",
            "rel_off_rtg",
            "rel_def_rtg",
            "rel_net_rtg",
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
        ]
    ]
'''
if __name__ == "__main__":
    base_dir = "Game Recaps"
    for csv_path in Path(base_dir).glob("*.csv"):
        game = csv_path.stem
        summary = process_lineups(base_dir=base_dir, games=[game])
        if summary is not None:
            output_path = f"Lineup Data/Game Lineups/lineup_summary_{game}.csv"
            summary.to_csv(output_path, index=False)
            print(f"Exported lineup summary to {output_path}")
'''

if __name__ == "__main__":
    games_to_include = None
    summary = process_lineups(base_dir="Game Recaps", games=games_to_include)
    if summary is not None:
        output_path = "Lineup Data/lineup_summary_noseattle.csv" if games_to_include is None else \
            f"Lineup Data/lineup_summary_{'_'.join(games_to_include)}.csv"
        summary.to_csv(output_path, index=False)
        print(f"Exported lineup summary to {output_path}")