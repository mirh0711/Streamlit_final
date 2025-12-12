## HELPER FUNCTIONS 
# utils.py
import pathlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------- CONFIG -----------------------------
# Folder where the model .pkl files live (adjust if needed)
BASE_DIR = pathlib.Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"


# ClASSIFICATION model paths
LOGREG_PATH = MODELS_DIR / "logistic_regression_full_features.pkl"
SVM_PATH = MODELS_DIR / "svm_full_features.pkl"
SCALER_PATH = MODELS_DIR / "scaler_full_features.pkl"

# List of input features expected by the CLASSIFICATION models.
CLASS_FEATURE_COLS = [
    "DC_POWER",
    "AC_POWER",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]

# === REGRESSION MODELS (DC power, plant-specific) ===
# Linear Regression (with scaler) and Random Forest for each plant
REG_LR_P1_PATH = MODELS_DIR / "lr_DC_P1.pkl"
REG_LR_P2_PATH = MODELS_DIR / "lr_DC_P2.pkl"
REG_SCALER_P1_PATH = MODELS_DIR / "scaler_DC_P1.pkl"
REG_SCALER_P2_PATH = MODELS_DIR / "scaler_DC_P2.pkl"

REG_RF_P1_PATH = MODELS_DIR / "rf_DC_P1.pkl"   # RF for Plant 1 (no scaler)
REG_RF_P2_PATH = MODELS_DIR / "rf_DC_P2.pkl"   # RF for Plant 2 (no scaler)


# Features the regression models expect (MUST match your training!)
REG_FEATURE_COLS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]

# Target column (these models predict DC power)
REG_TARGET_COL = "DC_POWER"


# === LSTM MODELS DATA LOADERS ===
LSTM_AC_P1_PATH    = MODELS_DIR / "LSTM_1h_Plant_1_AC_Power.pkl"
LSTM_AC_P2_PATH    = MODELS_DIR / "LSTM_1h_Plant_2_AC_Power.pkl"
LSTM_YIELD_P1_PATH = MODELS_DIR / "LSTM_1h_Plant_1_Yield.pkl"
LSTM_YIELD_P2_PATH = MODELS_DIR / "LSTM_1h_Plant_2_Yield.pkl"
LSTM_DC_P1_PATH    = MODELS_DIR / "LSTM_1h_DC_Power_Plant1.pkl"
LSTM_DC_P2_PATH    = MODELS_DIR / "LSTM_1h_DC_Power_Plant2.pkl"


# ---------- LSTM MODEL CLASS (for unpickling) ----------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get last time step output
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        return out

# ---------- HELPERS ----------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("Plant_1_and_2_unified.csv")
    # plant labels
    df["PLANT"] = df["PLANT_ID"].map({4135001: "Plant 1", 4136001: "Plant 2"})
    # binary label from Optimal_ratio (1 = optimal, 0 = suboptimal)
    df["optimal"] = np.where(df["Optimal_ratio"] >= 0.5, 1, 0)
    return df


def _load_single_model(path: pathlib.Path):
    """Try loading a model with joblib first, then pickle as a fallback."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# === CLASSIFICATION MODEL LOADER ===
@st.cache_resource
def load_models():
    """Classification models."""
    logreg = _load_single_model(LOGREG_PATH)   # best_pipe: SMOTE+Scaler+LogReg
    svm = _load_single_model(SVM_PATH)         # bare SVC trained on scaled data
    scaler = _load_single_model(SCALER_PATH)   # StandardScaler fitted on X_train

    # build a prediction pipeline: scaler -> svm
    svm_pipeline = Pipeline([
        ("scaler", scaler),
        ("svm", svm),
    ])

    return {
        "Logistic Regression": logreg,
        "Support Vector Machine": svm_pipeline,
    }



@st.cache_data
def get_regression_test_split_for_plant(plant_label: str):
    """Return (X_test, y_test) for the given plant, matching the notebook logic."""
    df_raw = pd.read_csv("Plant_1_and_2_unified.csv")

    # Same NaN filtering as notebook
    df_raw = df_raw.dropna(
        subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER"]
    )

    # Map label -> ID
    plant_id_map = {"Plant 1": 4135001, "Plant 2": 4136001}
    pid = plant_id_map[plant_label]

    plant_df = df_raw[df_raw["PLANT_ID"] == pid]

    X = plant_df[REG_FEATURE_COLS]
    y = plant_df[REG_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test


# === REGRESSION MODEL LOADER (plant-specific LR + RF + NN) ===
@st.cache_resource
def load_regression_models():
    """
    DC power regression models, plant-specific:

      Plant 1:
        - 'Plant 1 - LR' : scaler_DC_P1.pkl + lr_DC_P1.pkl
        - 'Plant 1 - RF' : rf_DC_P1.pkl

      Plant 2:
        - 'Plant 2 - LR' : scaler_DC_P2.pkl + lr_DC_P2.pkl
        - 'Plant 2 - RF' : rf_DC_P2.pkl

    The UI will show: 'Linear Regression', 'Random Forest';
    internally we map (plant, algorithm) → correct model.
    """
    models = {}

    def _load_lr(key_name, model_path, scaler_path):
        try:
            lr_model = _load_single_model(model_path)
            lr_scaler = _load_single_model(scaler_path)

            # Pipeline: StandardScaler (fitted in training) -> LinearRegression
            pipe = Pipeline([
                ("scaler", lr_scaler),
                ("lr", lr_model),
            ])
            models[key_name] = pipe
        except Exception as e:
            st.error(
                f"Could not load {key_name} "
                f"(model: {model_path.name}, scaler: {scaler_path.name}): {e}"
            )

    def _load_rf(key_name, model_path):
        try:
            rf_model = _load_single_model(model_path)
            models[key_name] = rf_model
        except Exception as e:
            st.error(
                f"Could not load {key_name} "
                f"(model: {model_path.name}): {e}"
            )

    # Plant 1 models
    _load_lr("Plant 1 - LR", REG_LR_P1_PATH, REG_SCALER_P1_PATH)
    _load_rf("Plant 1 - RF", REG_RF_P1_PATH)

    # Plant 2 models
    _load_lr("Plant 2 - LR", REG_LR_P2_PATH, REG_SCALER_P2_PATH)
    _load_rf("Plant 2 - RF", REG_RF_P2_PATH)

    return models



# --- Helper: pick the correct LR/RF model for the selected plant ---
def get_plant_specific_models(reg_models, plant_label):
    """Return (lr_model, rf_model) for the given plant name."""
    if plant_label == "Plant 1":
        lr_key = "Plant 1 - LR"
        rf_key = "Plant 1 - RF"
        #nn_key = None
    else:  # Plant 2
        lr_key = "Plant 2 - LR"
        rf_key = "Plant 2 - RF"
        #nn_key = "Plant 2 - NN"
    return reg_models.get(lr_key), reg_models.get(rf_key)


# === LSTM MODELS LOADER ===
@st.cache_resource
def load_lstm_models():
    """
    Load all LSTM temporal forecasting models, **plant-specific**:
      - AC Power  – Plant 1 / Plant 2
      - Yield     – Plant 1 / Plant 2
      - DC Power  – Plant 1 / Plant 2
    """
    models = {}

    def _safe_load(name: str, path: pathlib.Path):
        try:
            models[name] = _load_single_model(path)
        except Exception as e:
            st.error(f"Could not load LSTM model '{name}' from {path.name}: {e}")

    _safe_load("AC Power - Plant 1",   LSTM_AC_P1_PATH)
    _safe_load("AC Power - Plant 2",   LSTM_AC_P2_PATH)
    _safe_load("Yield - Plant 1",      LSTM_YIELD_P1_PATH)
    _safe_load("Yield - Plant 2",      LSTM_YIELD_P2_PATH)
    _safe_load("DC Power - Plant 1",   LSTM_DC_P1_PATH)
    _safe_load("DC Power - Plant 2",   LSTM_DC_P2_PATH)

    return models



def build_input_row(
    dc_power: float,
    ac_power: float,
    amb_temp: float,
    mod_temp: float,
    irradiation: float,
) -> pd.DataFrame:
    """Create a single-row DataFrame with the right feature columns for classification."""
    data = {
        "DC_POWER": dc_power,
        "AC_POWER": ac_power,
        "AMBIENT_TEMPERATURE": amb_temp,
        "MODULE_TEMPERATURE": mod_temp,
        "IRRADIATION": irradiation,
    }
    return pd.DataFrame([data], columns=CLASS_FEATURE_COLS)


# === REGRESSION INPUT BUILDER ===
def build_regression_input_row(amb_temp, mod_temp, irradiation) -> pd.DataFrame:
    """
    Single-row DataFrame for regression prediction.

    Columns are aligned with REG_FEATURE_COLS, which is set from the scaler’s
    feature_names_in_ inside load_regression_models().
    """
    data = {
        "AMBIENT_TEMPERATURE": amb_temp,
        "MODULE_TEMPERATURE": mod_temp,
        "IRRADIATION": irradiation,
    }
    return pd.DataFrame([data], columns=REG_FEATURE_COLS)



def plot_confusion_matrix(cm: np.ndarray, class_names):
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    # write counts on the cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    return fig


# Helper function to recreate train/test split for classification
@st.cache_data
def get_train_test(df: pd.DataFrame):
    # --- Rebuild Scenario A labels exactly as in the notebook ---
    ratio = df["Optimal_ratio"]
    irr = df["IRRADIATION"]

    cond_A = pd.Series(np.nan, index=df.index, dtype="float")
    known = ratio.notna()
    cond_A[known] = (ratio[known] > 0.5).astype(int)       # 1 = Optimal, 0 = Suboptimal
    unknown = ratio.isna()
    cond_A[unknown & (irr > 0.6)] = 1
    cond_A[unknown & (irr < 0.6)] = 0

    scenario_B_mask = ratio.notna()        # rows allowed for TEST
    valid_mask = cond_A.notna()            # rows where Scenario A defined
    y_A = cond_A[valid_mask].astype(int)

    # Build modelling DataFrame
    df_model = df.loc[valid_mask, CLASS_FEATURE_COLS].copy()
    df_model["y_A"] = y_A.values
    df_model["known_ratio"] = scenario_B_mask.loc[valid_mask].values

    # TEST: only known_ratio == True (Scenario B)
    eligible_test_idx = df_model.index[df_model["known_ratio"]]

    train_known_idx, test_idx = train_test_split(
        eligible_test_idx,
        test_size=0.2,
        stratify=df_model.loc[eligible_test_idx, "y_A"],
        random_state=42,
    )

    # TRAIN: remaining known rows + all unknown rows
    unknown_idx = df_model.index[~df_model["known_ratio"]]
    train_idx = train_known_idx.union(unknown_idx)

    X_train = df_model.loc[train_idx, CLASS_FEATURE_COLS]
    y_train = df_model.loc[train_idx, "y_A"]
    X_test = df_model.loc[test_idx, CLASS_FEATURE_COLS]
    y_test = df_model.loc[test_idx, "y_A"]

    return X_train, y_train, X_test, y_test

# ============= LSTM TEMPORAL FORECASTING HELPERS =============

def _prepare_lstm_base_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    Rebuild the same preprocessing as in 06b-Task6-LSTM-1H.ipynb:
    - group by DATE_TIME + PLANT_ID
    - create datetime features
    - fill NaNs for weather
    - MinMax-scale features + AC/DC/YIELD
    - split into Plant 1 / Plant 2
    """
    df = pd.read_csv(csv_path)

    # Group by datetime and plant (same as notebook)
    df = df.groupby(["DATE_TIME", "PLANT_ID"], as_index=False).agg(
        {
            "AC_POWER": "sum",
            "DC_POWER": "sum",
            "Optimal_ratio": "mean",
            "YIELD": "sum",
            "AMBIENT_TEMPERATURE": "mean",
            "MODULE_TEMPERATURE": "mean",
            "IRRADIATION": "mean",
        }
    )

    # Datetime index + time features
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df.set_index("DATE_TIME").sort_index()

    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    # Fill NaNs in weather columns
    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
        df[col] = df[col].interpolate(method="time").bfill().ffill()

    # Scaling like in the notebook
    feature_scaler = MinMaxScaler()
    ac_scaler = MinMaxScaler()
    dc_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    features_to_scale = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "minute",
        "hour",
        "day_of_week",
        "month",
    ]

    df[features_to_scale] = feature_scaler.fit_transform(df[features_to_scale])
    df["AC_POWER"] = ac_scaler.fit_transform(df[["AC_POWER"]])
    df["DC_POWER"] = dc_scaler.fit_transform(df[["DC_POWER"]])
    df["YIELD"] = y_scaler.fit_transform(df[["YIELD"]])

    # Split plants
    plant1 = df[df["PLANT_ID"] == 4135001].copy()
    plant2 = df[df["PLANT_ID"] == 4136001].copy()

    # Feature columns: remove targets + Optimal_ratio + PLANT_ID
    TARGET_LABEL = ["AC_POWER", "DC_POWER", "YIELD"]
    feature_columns = [
        c for c in df.columns if c not in TARGET_LABEL + ["Optimal_ratio", "PLANT_ID"]
    ]

    return {
        "df": df,
        "plant1": plant1,
        "plant2": plant2,
        "feature_columns": feature_columns,
        "feature_scaler": feature_scaler,
        "ac_scaler": ac_scaler,
        "dc_scaler": dc_scaler,
        "y_scaler": y_scaler,
        "SEQ_LENGTH": 4,   # 1h = 4 * 15min
        "TRAIN_SIZE": 0.8,
    }


def _create_lstm_sequences(features: np.ndarray, target: np.ndarray, seq_length: int):
    """
    Exactly like the notebook's create_sequences:
    X[t] = features[t : t+seq_length]
    y[t] = target[t+seq_length]
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i : i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)



@st.cache_data
def get_lstm_dc_power_plant1_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    DC power LSTM – PLANT 1 only.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant1 = base["plant1"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant1[feature_cols].to_numpy()
    t = plant1["DC_POWER"].to_numpy()

    X, y = _create_lstm_sequences(f, t, L)
    times = plant1.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["dc_scaler"],
    }




@st.cache_data
def get_lstm_dc_power_plant2_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    DC power LSTM – PLANT 2 only.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant2 = base["plant2"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant2[feature_cols].to_numpy()
    t = plant2["DC_POWER"].to_numpy()

    X, y = _create_lstm_sequences(f, t, L)
    times = plant2.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["dc_scaler"],
    }




@st.cache_data
def get_lstm_ac_power_plant1_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    AC power LSTM – PLANT 1 only.
    Same split logic as DC Plant 1.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant1 = base["plant1"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant1[feature_cols].to_numpy()
    t = plant1["AC_POWER"].to_numpy()   # scaled

    X, y = _create_lstm_sequences(f, t, L)
    times = plant1.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["ac_scaler"],
    }


@st.cache_data
def get_lstm_ac_power_plant2_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    AC power LSTM – PLANT 2 only.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant2 = base["plant2"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant2[feature_cols].to_numpy()
    t = plant2["AC_POWER"].to_numpy()

    X, y = _create_lstm_sequences(f, t, L)
    times = plant2.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["ac_scaler"],
    }


@st.cache_data
def get_lstm_yield_plant1_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    Yield LSTM – PLANT 1 only.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant1 = base["plant1"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant1[feature_cols].to_numpy()
    t = plant1["YIELD"].to_numpy()

    X, y = _create_lstm_sequences(f, t, L)
    times = plant1.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["y_scaler"],
    }


@st.cache_data
def get_lstm_yield_plant2_data(csv_path: str = "Plant_1_and_2_unified.csv"):
    """
    Yield LSTM – PLANT 2 only.
    """
    base = _prepare_lstm_base_data(csv_path)
    plant2 = base["plant2"]
    feature_cols = base["feature_columns"]
    L = base["SEQ_LENGTH"]
    train_size = base["TRAIN_SIZE"]

    f = plant2[feature_cols].to_numpy()
    t = plant2["YIELD"].to_numpy()

    X, y = _create_lstm_sequences(f, t, L)
    times = plant2.index[L:].values

    split = int(train_size * len(X))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "time_train": times[:split],
        "time_test": times[split:],
        "feature_columns": feature_cols,
        "feature_scaler": base["feature_scaler"],
        "target_scaler": base["y_scaler"],
    }




def compute_lstm_metrics_from_scaled(y_true_scaled, y_pred_scaled, scaler):
    """Inverse-transform y and compute MAE, RMSE, R² on original scale."""
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2, y_true, y_pred
