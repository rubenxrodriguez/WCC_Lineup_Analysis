import pandas as pd
import os
from pathlib import Path
import re

# ====================== CONFIGURATION ======================
# File paths
CONFERENCE_PLAYERS_CSV = 'wcc_players.csv'  # Contains fullName, teamMarket, playerId, height
TEAM_DATA_DIR = 'teams'                     # Directory containing team subfolders
SEASON_LINEUPS_FILE = 'top_lineups.csv'     # Name of season lineup files in each team folder
INTERVAL_FILE_FORMAT = 'games_{start}_{end}.csv'  # Format for interval files

# Analysis parameters
GAME_SPLITS = 10
TOTAL_GAMES = 30
TOP_LINEUPS_COUNT = 12

# ====================== DATA LOADING ======================
def load_player_data():
    """Load conference player data with height information"""
    player_df = pd.read_csv(CONFERENCE_PLAYERS_CSV)
    player_df['height'] = pd.to_numeric(player_df['height'], errors='coerce')
    
    # Create mapping dictionaries
    id_to_info = {}
    for _, row in player_df.iterrows():
        player_id = int(float(row['playerId']))
        parts = row['fullName'].split()
        initial = f"{parts[0][0]}{parts[-1][0:3]}".upper() if len(parts) >= 2 else row['fullName'][:2].upper()
        id_to_info[player_id] = {
            'initial': initial,
            'height': row['height'],
            'team': row['teamMarket']
        }
    return id_to_info

PLAYER_INFO = load_player_data()

def safe_int_convert(player_id):
    """Safely convert player ID to integer handling NaN/None"""
    try:
        return int(float(player_id))
    except (ValueError, TypeError):
        return None
    
def generate_game_intervals():
    """Create list of (start_game, end_game) tuples"""
    return [(i, i+GAME_SPLITS-1) for i in range(0, TOTAL_GAMES, GAME_SPLITS)]

def get_top_lineups(season_df):
    """Get top lineups excluding those with missing players"""
    return (season_df[season_df['lineup'].str.count('UNK') == 0]
            .sort_values('POSS', ascending=False)
            .head(TOP_LINEUPS_COUNT))
# ====================== CORE FUNCTIONS ======================
def create_height_sorted_lineup(row):
    """Create lineup string sorted by player height ascending"""
    players = []
    for col in ['pId1', 'pId2', 'pId3', 'pId4', 'pId5']:
        player_id = safe_int_convert(row[col])
        if player_id in PLAYER_INFO:
            info = PLAYER_INFO[player_id]
            players.append({
                'initial': info['initial'],
                'height': info['height'],
                'id': player_id
            })
        else:
            players.append({
                'initial': f"P{player_id}" if player_id else "UNK",
                'height': float('inf'),  # Sort unknown players last
                'id': player_id
            })
    
    # Sort by height then by initial
    sorted_players = sorted(players, key=lambda x: (x['height'], x['initial']))
    return '-'.join([p['initial'] for p in sorted_players])

def load_and_process_team_data(team_dir):
    """Load and process all data for a single team"""
    team_path = Path(TEAM_DATA_DIR) / team_dir
    
    # Load season data
    
    season_df = pd.read_csv(team_path / SEASON_LINEUPS_FILE, skiprows=1)
    
    
    season_df = process_dataframe(season_df)
    
    # Process intervals
    all_intervals = []
    for i, (start, end) in enumerate(generate_game_intervals(), 1):
        interval_file = team_path / INTERVAL_FILE_FORMAT.format(start=start+1, end=end+1)
        if interval_file.exists():
            interval_df = pd.read_csv(interval_file, skiprows = 1)
            interval_df = process_dataframe(interval_df)
            interval_df['interval'] = f'{start+1}-{end+1}'
            interval_df['interval_num'] = i
            all_intervals.append(interval_df)
    
    return season_df, pd.concat(all_intervals) if all_intervals else None

def process_dataframe(df):
    """Common processing for all dataframes"""
    # Convert numeric columns
    numeric_cols = ['POSS', 'MP*', 'Plus-Minus', 'Net Rtg', 'ORtg', 'DRtg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add calculated metrics
    df['Plus-Minus_per40'] = (df['Plus-Minus'] / df['MP*']) * 40
    
    # Create lineup string
    df['lineup'] = df.apply(create_height_sorted_lineup, axis=1)
    
    return df

# ====================== ANALYSIS FUNCTIONS ======================
def analyze_team(team_dir):
    """Full analysis pipeline for a single team"""
    season_df, interval_df = load_and_process_team_data(team_dir)
    
    if season_df is None:
        print(f"No data found for team {team_dir}")
        return None
    
    # Export top lineups
    top_lineups = get_top_lineups(season_df)
    export_top_lineups(top_lineups, team_dir)
    
    # Export progression data if available
    if interval_df is not None:
        export_progression_data(interval_df, team_dir)
    
    return season_df, interval_df

def export_top_lineups(df, team_name):
    """Export top lineups for a team"""
    output_metrics = {
        'lineup': 'lineup',
        'POSS': 'possessions',
        'MP*': 'minutes',
        'Plus-Minus': 'plusminus',
        'Plus-Minus_per40': 'plusminus_per40',
        'Net Rtg': 'netrating',
        'ORtg': 'offrating',
        'DRtg': 'defrating'
    }
    
    output_df = df[list(output_metrics.keys())].rename(columns=output_metrics)
    numeric_cols = output_df.select_dtypes(include=['float64', 'int64']).columns
    output_df[numeric_cols] = output_df[numeric_cols].round(2)
    
    output_file = f'output/{team_name}_top_lineups.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Top lineups saved to {output_file}")

def export_progression_data(df, team_name):
    """Export lineup progression data"""
    output_metrics = {
        'lineup': 'lineup',
        "POSS": 'possessions',
        "MP*" : 'minutes',
        "Plus-Minus" : "plusminus",
        "Plus-Minus_per40": "plusminus_per40",
        "Net Rtg": "netrating",
        "ORtg": "offrating",
        "DRtg": "defrating",
        'interval': 'interval',
        'interval_num': 'interval_num'
    }
    df  = df[list(output_metrics.keys())].rename(columns = output_metrics)
    
    output_cols = ['lineup', 'interval', 'interval_num', 'possessions', 
                   'minutes', 'plusminus_per40', 'netrating']
    output_file = f'output/{team_name}_progression.csv'
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    df[output_cols].to_csv(output_file, index=False)
    print(f"Progression data saved to {output_file}")

# ====================== EXECUTION ======================
if __name__ == '__main__':
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Get list of teams (subdirectories in TEAM_DATA_DIR)
    team_dirs = [d for d in os.listdir(TEAM_DATA_DIR) 
                if os.path.isdir(os.path.join(TEAM_DATA_DIR, d))]
    
    # Analyze all teams
    for team in team_dirs:
        print(f"\nAnalyzing team: {team}")
        analyze_team(team)
    
    print("\nAnalysis complete for all teams!")