import pandas as pd
from itertools import combinations

def _safe_div(numer, denom):
    return numer / denom if denom else 0

PLAYER_INFO = {
    "JL": {"pid": "2138783", "name": "Jess Lawson", "initial": "JL", "height": 67},
    "MS": {"pid": "2118049", "name": "Mari Somvichian", "initial": "MS", "height": 64},
    "AM": {"pid": "1925105", "name": "Andjela Matic", "initial": "AM", "height": 69},
    "IK": {"pid": "2270840", "name": "Ivana Krajina", "initial": "IK", "height": 71},
    "ZO": {"pid": "2291264", "name": "Zawadi Ogot", "initial": "ZO", "height": 71},
    "LL": {"pid": "2283482", "name": "Lova Lagerlid", "initial": "LL", "height": 73},
    "CH": {"pid": "1688590", "name": "Carly Heidger", "initial": "CH", "height": 75},
    "KJ": {"pid": "2481512", "name": "Kayla Jones", "initial": "KJ", "height": 72},
    "MH": {"pid": "2118064", "name": "Maya Hernandez", "initial": "MH", "height": 72},
    "AM7": {"pid": "2283300", "name": "Ana Milanovic", "initial": "AM7", "height": 75},
    "PR": {"pid": "1696172", "name": "Paula Reus Piza", "initial": "PR", "height": 74},
    "AC": {"pid": "2291259", "name": "Allison Clarke", "initial": "AC", "height": 71},
}

def sort_players_by_height(players, player_info):
    """
    players: iterable of player identifiers (initials or pids)
    Returns tuple sorted by height DESC, then identifier ASC.
    """

    def sort_key(p):
        info = player_info.get(p)
        height = info["height"] if info else -1
        return (height, p)

    return tuple(sorted(players, key=sort_key))


def analyze_combos(
    lineup_summary_path="Lineup Data/lineup_summary_inconf12.csv",
    output_path="Lineup Data/pair_analysis_inconf12.csv",
    combo_size=3,
    min_minutes=15,
):
    """
    Build N-player combo metrics from a lineup summary table produced by updated_lineups.py.
    combo_size=2 -> pairs, combo_size=3 -> trios, etc.
    """
    if combo_size < 2 or combo_size > 5:
        raise ValueError("combo_size must be between 2 and 5 (lineups have 5 players).")

    df = pd.read_csv(lineup_summary_path)
    if df.empty:
        print("Lineup summary is empty; no combos to analyze.")
        return None

    rows = []
    for _, row in df.iterrows():
        players = row["lineup"].split("-")

        # combos(players, 3) yields tuples like ('JL','MH','AM')
        for combo in combinations(players, combo_size):
            # canonical ordering so the same trio groups together even if lineup order changes
            combo = sort_players_by_height(combo, PLAYER_INFO)

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
        print("No combos found.")
        return None

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

    # “Team reference” using the same approach as your pair file
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

    combo_stats = combo_stats.round(3)
    if min_minutes is not None:
        combo_stats = combo_stats[combo_stats["minutes"] >= min_minutes].copy()

    # nice-to-have sort
    combo_stats = combo_stats.sort_values(["minutes", "net_rtg"], ascending=[False, False])

    combo_stats.to_csv(output_path, index=False)
    print(f"Exported {combo_size}-player combo analysis to {output_path}")
    return combo_stats

if __name__ == "__main__":
    # 3-player combos
    result = analyze_combos(combo_size=2,min_minutes =20)
    if result is not None:
        print(result.head())
