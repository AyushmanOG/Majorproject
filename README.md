# Majorproject
## IPL Batsman Performance Analysis and Clustering (2008–2020)

## 📝 Project Title
IPL Batsman Performance Analysis and Clustering (2008–2020)

## 📌 Objective
To analyze IPL ball-by-ball data from 2008–2020 and use **unsupervised machine learning** techniques to:
- Understand batsmen’s performance
- Identify the top-performing players
- Group similar batsmen into performance-based clusters

## 🗂️ Dataset Used
- **File**: `IPL Ball-by-Ball 2008-2020.csv`
- **Features include**:
  - `match_id`, `batsman`, `bowler`, `batting_team`, `bowling_team`
  - `total_runs`, `batsman_runs`, `is_wicket`, `dismissal_kind`, etc.
- The data provides detailed per-ball information for over a decade of IPL.

## 🔧 Data Preprocessing
1. **Renaming & Cleaning**:
   - Renamed column `id` to `match_id`.
   - Standardized team names (e.g., `Delhi Daredevils` → `Delhi Capitals`).
   - Removed or replaced missing values in:
     - `dismissal_kind`, `player_dismissed`, `extras_type`, `bowling_team`.

2. **Null Handling**:
   - Filled `dismissal_kind` with `'not out'` when `is_wicket == 0`.
   - Dropped rows where `bowling_team` was missing.

## 📊 Exploratory Data Analysis (EDA)

### Team-Level Analysis
- Calculated **total runs** by each team.
- Plotted a **pie chart** of the **top 5 scoring teams**.

### Player-Level Analysis
- Identified **top 10 batsmen** by total runs.
- Identified **top 10 bowlers** by total wickets.
- Identified **top 10 fielders** by number of dismissals involved.
- Visualized the results using bar charts.

## 📈 Feature Engineering
For each batsman:
- **matches_played**
- **total_runs**
- **balls_faced**
- **strike_rate**
- **number of sixes**
- **number of fours**

This dataset was used for clustering.

## 🚨 Outlier Detection
- Used **Isolation Forest** to identify anomalous players based on performance.
- Removed about 5% of data points considered as outliers.

## ⚙️ Feature Scaling
- Applied **StandardScaler** to normalize data before clustering.

## 🧠 Clustering Techniques

### 1. KMeans Clustering
- Found optimal cluster count using **Elbow Method**.
- Clustered batsmen into 4 performance groups.
- Evaluated using:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Score
- **Visualized clusters** using:
  - Radar Charts
  - Histograms of feature distributions

### 2. Hierarchical Clustering (Agglomerative)
- Created a **dendrogram** to understand data structure.
- Chose 4 clusters using linkage method = `ward`.
- Similar evaluation metrics were used.
- Radar charts and histograms again illustrated cluster profiles.

### 3. DBSCAN (Density-Based Spatial Clustering)
- Determined epsilon value using **k-nearest neighbors distance graph**.
- Clustered batsmen into core and noise groups.
- Evaluated performance using the same metrics.

## 🔍 Evaluation of Clustering Models
Each algorithm’s clustering effectiveness was evaluated using:
- **Silhouette Score**
- **Calinski-Harabasz Score**
- **Davies-Bouldin Score**

A comparison bar chart was plotted for **model selection**.

## 🏏 Identifying the Best Batsmen
- Selected batsmen with:
  - **≥ 100 matches**
  - **≥ 2500 total runs**
- Highlighted their **assigned clusters** under each model
- Exported this data to `ipl_top_batsmen_stats 2008-2020.csv`

## 📈 Visualizations
You included an impressive set of charts:
- **Bar Plots**, **Pie Charts**, **Histograms**
- **Radar Charts**, **Dendrograms**, **Model Comparison Charts**

## 📦 Output
- Final cleaned and clustered batsman performance data
- Exported as: `ipl_top_batsmen_stats 2008-2020.csv`
- All visualizations created inline

## 🛠️ Technologies Used
- Python 3
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `tabulate`, `scipy`

## ✅ How to Run

1. Ensure `IPL Ball-by-Ball 2008-2020.csv` is in your working directory.
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tabulate
   ```
3. Run the script using:
   ```bash
   python Major project.py
   ```
