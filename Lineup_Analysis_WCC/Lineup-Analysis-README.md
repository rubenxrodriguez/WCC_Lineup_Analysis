# WCC Lineup Analysis

This project provides a comprehensive analytical framework for evaluating lineup performance and stability across teams in the West Coast Conference (WCC). It utilizes +/- per 40 min and net rating to assess lineup effectiveness, stability over time, and performance metrics, offering valuable insights for coaches, analysts, and basketball enthusiasts.
### ** Notes **
* Thresholds were arbitrarily chosen
* There is a bias towards practicality over mathematical completeness
* The lineup data is from "cbbanalytics.com"
    * Because of the structure of the online data, I decided to pull 4 different time intervals and one cumulative end-of-season file
         * Interval 1 [start of season - Dec 15] , Interval 2 [Dec 16 - Jan 15], Interval 3 [Jan 16 - Feb 15], Interval 4 [Feb 16 - end of season]

## 📊 Features

- **Lineup Filtering**: Isolate top-performing lineups based on custom criteria.
- **Performance Metrics**: Compute key statistics such as net rating, plus-minus per 40 minutes, and their respective standard deviations.
- **Stability Analysis**: Determine the point at which a lineup's performance stabilizes using cumulative metrics.
- **Visualizations**: Generate insightful plots to visualize minutes distribution, performance stability, and stabilization timelines.
- **Interactive Tables**: Highlight lineups exceeding specified thresholds in minutes played or performance variability.


## ⚙️ Functions Overview

- **`load_all_team_data(teams_list)`**: Loads progression and top lineup data for specified teams.
- **`analyze_conference(progression_df, top_lineups_df)`**: Performs comprehensive analysis, returning stability statistics and minimum sample sizes for stabilization.
- **`visualize_conference_trends(stats_df, min_samples_df)`**: Generates visualizations for minutes distribution, performance stability, and stabilization timelines.

## 📈 Methodology

- **Stability Metrics**: Calculates the coefficient of variation (CV) for net rating and plus-minus per 40 minutes to assess performance consistency.
- **Stabilization Point**: Identifies the cumulative minutes at which a lineup's performance metric change falls below a defined threshold, indicating stabilization.
- **Weighted Averages**: Ensures accurate representation of performance metrics by computing weighted averages based on minutes played.


## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rubenxrodriguez/WCC_Lineup_Analysis.git
   cd WCC_Lineup_Analysis/Lineup_Analysis_WCC
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Directory Structure

```
Lineup_Analysis_WCC/
├── data/                   # Raw and processed data files
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Python scripts for data processing and analysis
├── visuals/                # Generated plots and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Usage

1. **Load Data**:
   ```python
   from scripts.data_loader import load_all_team_data

   teams_list = ['TeamA', 'TeamB', 'TeamC']  # Replace with actual team names
   wcc_progression, wcc_top_lineups = load_all_team_data(teams_list)
   ```

2. **Analyze Conference Data**:
   ```python
   from scripts.analysis import analyze_conference

   stability_stats, min_samples_df = analyze_conference(wcc_progression, wcc_top_lineups)
   ```

3. **Visualize Trends**:
   ```python
   from scripts.visualization import visualize_conference_trends

   visualize_conference_trends(stability_stats, min_samples_df)
   ```

4. **Highlight Specific Lineups**:
   ```python
   import pandas as pd

   delta = 50  # Threshold in minutes
   lineups_over_50 = wcc_progression[wcc_progression['minutes'] > delta][
       ['lineup', 'team', 'interval_num', 'minutes', 'plusminus_per40']
   ]
   pd.set_option('display.max_rows', None)
   display(lineups_over_50.style.background_gradient(cmap='coolwarm', subset=['plusminus_per40']))
   ```

Ensure that all dependencies are installed and that you're in the correct directory.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📬 Contact

For questions or collaboration inquiries, please contact [Ruben Rodriguez](mailto:rrodr102@lion.lmu.edu).
