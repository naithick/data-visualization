# Marketing Campaign Data Visualization & Optimization

This repository contains a comprehensive analysis of marketing campaign performance, including data cleaning, exploratory data analysis (EDA), segmentation, and ROI prediction using Machine Learning.

## 📂 Project Structure

```
.
├── data
│   ├── raw             # Original dataset
│   └── processed       # Cleaned and enriched data
├── models              # Serialized ML models and encoders
├── results
│   ├── figures         # Generated plots and visualizations
│   └── reports         # Analysis reports
├── src                 # Source code for analysis and optimization
├── .gitignore          # Git ignore file
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/naithick/data-visualization.git
    cd data-visualization
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### Running the Optimizer

The main script `src/optimizer.py` performs the entire pipeline:
1.  **Data Cleaning**: Handles missing values and outliers.
2.  **Feature Engineering**: Creates metrics like CPA, ROAS, and Engagement Yield.
3.  **Segmentation**: Clusters campaigns into "Whales", "Hidden Gems", "Money Pits", and "Standard".
4.  **Modeling**: Trains a Gradient Boosting Regressor to predict ROI.
5.  **Visualization**: Generates insight charts in `results/figures/`.
6.  **Simulation**: Runs budget allocation scenarios.

Run the script from the project root:

```bash
python src/optimizer.py
```

*Note: The script outputs charts to `results/figures/` and processed data to `data/processed/`.*

## 📈 Key Insights

-   **Whales**: High ROI & High Cost campaigns. Scale these carefully.
-   **Hidden Gems**: High ROI & Low Cost. Prime candidates for budget increases.
-   **Money Pits**: Low ROI & High Cost. These should be paused or optimized.
-   **Saturation**: Analysis shows diminishing returns beyond certain spend thresholds.

## 🛠 Technologies

-   **Pandas & NumPy**: Data manipulation
-   **Matplotlib & Seaborn**: Visualization
-   **Scikit-learn**: Machine Learning (K-Means, Gradient Boosting)
-   **Joblib**: Model persistence

## 📜 Deliverables

-   **Source Code**: `src/optimizer.py` covers the full analysis pipeline.
-   **Visualizations**: See `results/figures` for generated plots.
-   **Data**: Processed datasets in `data/processed`.
