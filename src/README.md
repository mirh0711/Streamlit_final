# Project README

# Solar Plant Dashboard (Streamlit)

This Streamlit app is an interactive dashboard for exploring solar power data from two plants, forecasting DC power, forecasting time series with LSTMs, and classifying whether a plant is operating optimally.

The app uses pre-trained models (saved as `.pkl` files) together with the unifi
ed dataset `Plant_1_and_2_unified.csv`.

---

## 1. Project structure

```text
project-root/
│
├─ src/
│  ├─ app.py            # Main Streamlit application
│  ├─ utils.py          # Helper functions (data loading, models, LSTM helpers, etc.)
│  └─ README.md         # This file
│
├─ Models/              # Pre-trained models (must exist for the app to work)
│  ├─ logistic_regression_full_features.pkl
│  ├─ svm_full_features.pkl
│  ├─ scaler_full_features.pkl
│  ├─ lr_DC_P1.pkl
│  ├─ lr_DC_P2.pkl
│  ├─ scaler_DC_P1.pkl
│  ├─ scaler_DC_P2.pkl
│  ├─ rf_DC_P1.pkl
│  ├─ rf_DC_P2.pkl
│  ├─ LSTM_1h_Plant_1_AC_Power.pkl
│  ├─ LSTM_1h_Plant_2_AC_Power.pkl
│  ├─ LSTM_1h_Plant_1_Yield.pkl
│  ├─ LSTM_1h_Plant_2_Yield.pkl
│  ├─ LSTM_1h_DC_Power_Plant1.pkl
│  └─ LSTM_1h_DC_Power_Plant2.pkl
│
└─ Plant_1_and_2_unified.csv  # Input data used by the app
```
Important: Plant_1_and_2_unified.csv and the Models/ folder should be at the paths expected above relative to src/app.py.

## 2. Installation

Create and activate a conda environment:

conda create -n solar-dashboard python=3.11 -y
conda activate solar-dashboard

Install dependencies: pip install -r requirements.txt
Otherwise, the key packages are:  pip install streamlit numpy pandas scikit-learn matplotlib altair torch


## 3. Running Application 
Make sure the following are in the same folder as app.py / app_try.py:
- Plant_1_and_2_unified.csv
- The Models/ directory with all .pkl files
- The paths inside utils.py assume:
.
├── app.py
├── utils.py
├── Plant_1_and_2_unified.csv
└── Models/
    └── *.pkl

If your structure is different, update BASE_DIR / MODELS_DIR in utils.py accordingly.

From the project root (the folder containing src/):  streamlit run src/app.py

Streamlit will open the app in your browser (or give you a local URL, usually http://localhost:8501).


## 4. Overview of the interface

The app has four main tabs:

### 4.1 Data overview

- Filter by plant (`Plant 1` / `Plant 2`) using the sidebar.
- Choose any two columns as the x- and y-axes.
- Interactive scatter plot coloured by plant.
- Summary statistics for all numeric columns (mean, std, quartiles, etc.).

![Tab 1 graph](dashboard_screenshots/tab1_graph.png)
![Tab 1 summary stats](dashboard_screenshots/tab1_stats.png)
---

### 4.2 DC Power forecasting

This tab focuses on **regression models for DC power**.

**Inputs**

- Select **Plant 1** or **Plant 2**.
- Adjust three environmental sliders:
  - `AMBIENT_TEMPERATURE`
  - `MODULE_TEMPERATURE`
  - `IRRADIATION`

**Single-point prediction**

- The app displays **Linear Regression** and **Random Forest** predictions side by side for the selected environmental conditions.
- Models are plant-specific:
  - Linear Regression uses a plant-specific scaler + model.
  - Random Forest uses plant-specific models without scaling.

**Historical performance**

For the selected plant, the app:

- Rebuilds a plant-specific train/test split (matching the notebook logic).
- Computes performance metrics:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² (coefficient of determination)**
- Shows scatter plots of **Actual vs Predicted** DC power for:
  - Linear Regression  
  - Random Forest  

This allows users to:
- Compare accuracy of the two models on historical data.
- See how point predictions change as the sliders are adjusted.

![Tab 2 – DC power forecasting](dashboard_screenshots/tab2_overview.png)
---

### 4.3 Optimal Ratio Classification

This tab predicts whether a time step is operating **Optimally** or **Suboptimally**.

**Inputs**

- Sliders for:
  - `DC_POWER`
  - `AC_POWER`
  - `AMBIENT_TEMPERATURE`
  - `MODULE_TEMPERATURE`
  - `IRRADIATION`

**Models**

- Logistic Regression
- Support Vector Machine (SVM) wrapped in a pipeline with a `StandardScaler`.

![Tab 3 graph](dashboard_screenshots/tab3_graph.png)

**Features**

- Per-input prediction with the selected model or an option to compare both.
- Class probabilities (bar chart) when `predict_proba` is available.
- Performance evaluation on the Scenario B test set:
  - Classification report (precision, recall, F1-score).
  - Confusion matrix plot.

The Scenario A/B labelling rules from the notebook are re-implemented in `utils.py` so the app uses consistent labels.
![Tab 3 stats](dashboard_screenshots/tab3_stats.png)
---

### 4.4 Temporal forecasting (LSTM)

This tab uses pre-trained **LSTM models** to forecast power and yield over time.

**Choices**

- Plant: `Plant 1` or `Plant 2`
- Target variable:
  - `AC_POWER`
  - `DC_POWER`
  - `YIELD`

**Preprocessing (mirrors the notebook)**

- Aggregate by `DATE_TIME` and `PLANT_ID`.
- Add time-based features: minute, hour, day of week, month.
- Interpolate and scale features and targets with `MinMaxScaler`.
- Build input sequences of length 4 (1 hour of 15-minute intervals).

**Visualisation**

- Time-series plots of **Actual vs Predicted** values over a user-selectable date range.
- A summary table of LSTM test-set metrics:
  - MAE, RMSE, R² for:
    - AC Power – Plant 1 & Plant 2
    - DC Power – Plant 1 & Plant 2
    - Yield – Plant 1 & Plant 2

![Tab 4 graph](dashboard_screenshots/tab4_graph.png)
![Tab 4 stats](dashboard_screenshots/tab4_stats.png)


