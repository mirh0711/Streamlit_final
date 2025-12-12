# STREAMLIT APP FOR SOLAR PLANT DASHBOARD
import pathlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import torch 
import torch.nn as nn


# import helpers from utils.py
from utils import (
    load_data,
    load_models,
    load_regression_models,
    load_lstm_models,
    build_input_row,
    build_regression_input_row,
    plot_confusion_matrix,
    get_train_test,
    get_regression_test_split_for_plant,
    get_plant_specific_models,
    get_lstm_ac_power_plant1_data,
    get_lstm_ac_power_plant2_data,
    get_lstm_yield_plant1_data,
    get_lstm_yield_plant2_data,
    get_lstm_dc_power_plant1_data,
    get_lstm_dc_power_plant2_data,
    CLASS_FEATURE_COLS,
    REG_FEATURE_COLS,
    REG_TARGET_COL,
    LSTMRegressor,
    compute_lstm_metrics_from_scaled,
)

# ========================================= MAIN APP =========================================


st.set_page_config(page_title="Group 9 Dashboard",
                   layout="wide")

df = load_data()
class_models = load_models()            # renamed from 'models'
reg_models = load_regression_models()
lstm_models = load_lstm_models()        # NEW: load LSTM temporal models

# Classification train/test split (Scenario B)
X_clf_train, y_clf_train, X_clf_test, y_clf_test = get_train_test(df)


st.title("Group 9 - Solar Plants Dashboard")
st.write(
    "Explore the solar plant data, forecast power output with regression models, "
    "forecast time series with LSTM models, and test the trained classification "
    "models that detect optimal vs suboptimal operation."
)

# Sidebar filters used across tabs
st.sidebar.header("Filters")
plant_options = df["PLANT"].unique().tolist()
selected_plants = st.sidebar.multiselect(
    "Select plant(s)", options=plant_options, default=plant_options
)

filtered_df = df[df["PLANT"].isin(selected_plants)]

x_axis = st.sidebar.selectbox(
    "X-axis feature (overview tab)", options=filtered_df.columns.tolist(), index=0
)
y_axis = st.sidebar.selectbox(
    "Y-axis feature (overview tab)", options=filtered_df.columns.tolist(), index=1
)

# --- TABS: add LSTM tab between power forecasting and classification ---
tab_overview, tab_forecast, tab_class, tab_lstm = st.tabs(
    ["Data overview", "DC Power forecasting", "Optimal Ratio Classification", "Temporal forecasting (LSTM)"]
)


# ----- TAB 1: DATA OVERVIEW -----
with tab_overview:
    st.subheader("Scatter plot")

    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(size=15)
        .encode(
            x=x_axis,
            y=y_axis,
            color="PLANT",
            tooltip=list(filtered_df.columns),
        )
        .interactive()
    )
    st.altair_chart(scatter, width='stretch')

    st.subheader("Summary statistics (numeric columns)")
    st.write(filtered_df.select_dtypes(include=["number"]).describe())




# ----- TAB 2: POWER FORECASTING (REGRESSION) -----
with tab_forecast:
    st.subheader("Power forecasting (regression models)")

    if not reg_models:
        st.info(
            "No regression models could be loaded. "
            "Check lr_DC_P1/2.pkl, rf_DC_P1/2.pkl and scaler_DC_P1/2.pkl "
            "exist in the Models folder."
        )
    else:
        left_col, right_col = st.columns([1, 2])

        # ---------------- LEFT: INPUTS ----------------
        with left_col:
            st.markdown("### 1. Set environmental conditions")

            # Choose plant (this drives which model we use)
            plant_options_forecast = df["PLANT"].unique().tolist()  # ['Plant 1', 'Plant 2']
            selected_plant_forecast = st.selectbox(
                "Plant (for ranges)",
                plant_options_forecast,
            )

            plant_df = df[df["PLANT"] == selected_plant_forecast]
            num_df_plant = plant_df.select_dtypes(include=["number"])

            amb_temp = st.slider(
                "AMBIENT_TEMPERATURE (°C)",
                float(num_df_plant["AMBIENT_TEMPERATURE"].min()),
                float(num_df_plant["AMBIENT_TEMPERATURE"].max()),
                float(num_df_plant["AMBIENT_TEMPERATURE"].median()),
            )
            mod_temp = st.slider(
                "MODULE_TEMPERATURE (°C)",
                float(num_df_plant["MODULE_TEMPERATURE"].min()),
                float(num_df_plant["MODULE_TEMPERATURE"].max()),
                float(num_df_plant["MODULE_TEMPERATURE"].median()),
            )
            irradiation = st.slider(
                "IRRADIATION",
                float(num_df_plant["IRRADIATION"].min()),
                float(num_df_plant["IRRADIATION"].max()),
                float(num_df_plant["IRRADIATION"].median()),
            )

            # Input row for LR/RF (3 environmental features only)
            reg_input_df = build_regression_input_row(
                amb_temp=amb_temp,
                mod_temp=mod_temp,
                irradiation=irradiation,
            )

        # ---------------- RIGHT: PREDICTION + METRICS ----------------
        with right_col:
            # Get plant-specific LR and RF once (NN is separate)
            lr_model, rf_model = get_plant_specific_models(
                reg_models, selected_plant_forecast
            )

            target_label = "DC power" if REG_TARGET_COL == "DC_POWER" else REG_TARGET_COL

            # ===== 2. Single-point predictions (LR & RF side by side) =====
            st.markdown("### 2. Predicted power output (single point)")

            sp_lr_col, sp_rf_col = st.columns(2)

            # ---- Linear Regression single point ----
            with sp_lr_col:
                st.markdown("**Linear Regression**")
                if lr_model is None:
                    st.warning(
                        f"Linear Regression model for {selected_plant_forecast} "
                        "is not available."
                    )
                else:
                    try:
                        y_pred_single_lr = lr_model.predict(reg_input_df)[0]
                        st.metric(
                            f"Predicted {target_label} ({selected_plant_forecast})",
                            f"{y_pred_single_lr:,.2f}",
                        )
                    except Exception as e:
                        st.error(f"Error when predicting with Linear Regression: {e}")

            # ---- Random Forest single point ----
            with sp_rf_col:
                st.markdown("**Random Forest**")
                if rf_model is None:
                    st.warning(
                        f"Random Forest model for {selected_plant_forecast} "
                        "is not available."
                    )
                else:
                    try:
                        y_pred_single_rf = rf_model.predict(reg_input_df)[0]
                        st.metric(
                            f"Predicted {target_label} ({selected_plant_forecast})",
                            f"{y_pred_single_rf:,.2f}",
                        )
                    except Exception as e:
                        st.error(f"Error when predicting with Random Forest: {e}")

            # ===== 3. Historical performance: LR, RF, NN =====
            st.markdown("### 3. Model performance on historical data")

            # Evaluate ONLY on this plant's data so we never mix Plant 1 & 2.
            X_eval, y_eval = get_regression_test_split_for_plant(selected_plant_forecast)

            if X_eval.empty:
                st.info("Not enough clean data for performance metrics for this plant.")
            else:
                col_lr, col_rf = st.columns(2)

                # ---- Linear Regression (plant-specific) ----
                with col_lr:
                    st.markdown("**Linear Regression**")
                    if lr_model is None:
                        st.warning(
                            f"Linear Regression model for {selected_plant_forecast} "
                            "is not available."
                        )
                    else:
                        try:
                            y_hat_lr = lr_model.predict(X_eval)

                            mae_lr = mean_absolute_error(y_eval, y_hat_lr)
                            rmse_lr = root_mean_squared_error(y_eval, y_hat_lr)
                            r2_lr = r2_score(y_eval, y_hat_lr)

                            st.write(
                                f"MAE: {mae_lr:,.2f} &nbsp;&nbsp; "
                                f"RMSE: {rmse_lr:,.2f} &nbsp;&nbsp; "
                                f"R²: {r2_lr:.3f}",
                                unsafe_allow_html=True,
                            )

                            plot_df_lr = pd.DataFrame(
                                {"Actual": y_eval.values, "Predicted": y_hat_lr}
                            )
                            reg_scatter_lr = (
                                alt.Chart(plot_df_lr)
                                .mark_circle(size=10)
                                .encode(
                                    x="Actual:Q",
                                    y="Predicted:Q",
                                    tooltip=["Actual", "Predicted"],
                                )
                                .properties(height=300)
                                .interactive()
                            )
                            st.altair_chart(reg_scatter_lr, width="stretch")
                        except Exception as e:
                            st.warning(
                                f"Error evaluating Linear Regression "
                                f"for {selected_plant_forecast}: {e}"
                            )

                # ---- Random Forest (plant-specific) ----
                with col_rf:
                    st.markdown("**Random Forest**")
                    if rf_model is None:
                        st.warning(
                            f"Random Forest model for {selected_plant_forecast} "
                            "is not available."
                        )
                    else:
                        try:
                            y_hat_rf = rf_model.predict(X_eval)

                            mae_rf = mean_absolute_error(y_eval, y_hat_rf)
                            rmse_rf = root_mean_squared_error(y_eval, y_hat_rf)
                            r2_rf = r2_score(y_eval, y_hat_rf)

                            st.write(
                                f"MAE: {mae_rf:,.2f} &nbsp;&nbsp; "
                                f"RMSE: {rmse_rf:,.2f} &nbsp;&nbsp; "
                                f"R²: {r2_rf:.3f}",
                                unsafe_allow_html=True,
                            )

                            plot_df_rf = pd.DataFrame(
                                {"Actual": y_eval.values, "Predicted": y_hat_rf}
                            )
                            reg_scatter_rf = (
                                alt.Chart(plot_df_rf)
                                .mark_circle(size=10)
                                .encode(
                                    x="Actual:Q",
                                    y="Predicted:Q",
                                    tooltip=["Actual", "Predicted"],
                                )
                                .properties(height=300)
                                .interactive()
                            )
                            st.altair_chart(reg_scatter_rf, width="stretch")
                        except Exception as e:
                            st.warning(
                                f"Error evaluating Random Forest "
                                f"for {selected_plant_forecast}: {e}"
                            )




# ----- TAB 3: TEMPORAL FORECASTING (LSTM) -----
with tab_lstm:
    st.subheader("Temporal Forecasting with LSTM")

    if not lstm_models:
        st.warning("No LSTM temporal models could be loaded. Check the .pkl files in the Models folder.")
    else:
        # Helper for calling LSTM models (sklearn style, torch, or generic callable)
        
        def lstm_predict(model, X):
            """
            X: numpy array of shape (n_samples, seq_len, n_features)
            Returns: numpy array of shape (n_samples,)
            """
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = model(X_tensor)              # shape (n_samples, 1)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.squeeze(-1).cpu().numpy()
            return outputs.reshape(-1)


        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.markdown("### 1. Select plant and target")

            plant_choice = st.selectbox(
                "Plant",
                ["Plant 1", "Plant 2"],
                key="lstm_plant_select",
            )

            target_choice = st.radio(
                "Forecast variable",
                ["AC Power", "DC Power", "Yield"],
                index=0,
            )

            # Decide which LSTM model and helper to use (all plant-specific)
            if target_choice == "AC Power":
                if plant_choice == "Plant 1":
                    model_key = "AC Power - Plant 1"
                    data_func = get_lstm_ac_power_plant1_data
                else:
                    model_key = "AC Power - Plant 2"
                    data_func = get_lstm_ac_power_plant2_data
                ylabel = "AC Power"

            elif target_choice == "Yield":
                if plant_choice == "Plant 1":
                    model_key = "Yield - Plant 1"
                    data_func = get_lstm_yield_plant1_data
                else:
                    model_key = "Yield - Plant 2"
                    data_func = get_lstm_yield_plant2_data
                ylabel = "Yield"

            else:  # DC Power
                if plant_choice == "Plant 1":
                    model_key = "DC Power - Plant 1"
                    data_func = get_lstm_dc_power_plant1_data
                else:
                    model_key = "DC Power - Plant 2"
                    data_func = get_lstm_dc_power_plant2_data
                ylabel = "DC Power"



            if model_key not in lstm_models:
                st.error(f"LSTM model '{model_key}' is not available.")
                st.stop()

            lstm_model = lstm_models[model_key]
            data = data_func()  # cached

            X_lstm_test = data["X_test"]
            y_lstm_test_scaled = data["y_test"]
            time_lstm_test = pd.to_datetime(data["time_test"])
            target_scaler = data["target_scaler"]




        with right_col:
            st.markdown("### 2. Forecast vs actual over time")

            # Predictions (scaled)
            try:
                y_lstm_pred_scaled = lstm_predict(lstm_model, X_lstm_test)
                # Clip to [0, 1] because MinMaxScaler was fit on that range
                #y_lstm_pred_scaled = np.clip(y_lstm_pred_scaled, 0.0, 1.0)

            except Exception as e:
                st.error(f"Error calling LSTM model '{model_key}': {e}")
                st.stop()

            # Inverse-transform to original scale
            y_lstm_true = target_scaler.inverse_transform(
                y_lstm_test_scaled.reshape(-1, 1)
            ).ravel()
            y_lstm_pred = target_scaler.inverse_transform(
                y_lstm_pred_scaled.reshape(-1, 1)
            ).ravel()

            # Time range slider
            # Convert to native Python datetime for Streamlit
            min_time = time_lstm_test.min()
            max_time = time_lstm_test.max()
            if isinstance(min_time, pd.Timestamp):
                min_time = min_time.to_pydatetime()
            if isinstance(max_time, pd.Timestamp):
                max_time = max_time.to_pydatetime()

            time_range = st.slider(
                "Select time range",
                min_value=min_time,
                max_value=max_time,
                value=(min_time, max_time),
                format="YYYY-MM-DD HH:mm",
            )

            mask = (time_lstm_test >= time_range[0]) & (time_lstm_test <= time_range[1])
            time_sel = time_lstm_test[mask]
            y_lstm_true_sel = y_lstm_true[mask]
            y_lstm_pred_sel = y_lstm_pred[mask]


            plot_df = pd.DataFrame(
                {
                    "DATE_TIME": np.concatenate([time_sel, time_sel]),
                    "Value": np.concatenate([y_lstm_true_sel, y_lstm_pred_sel]),
                    "Type": ["Actual"] * len(time_sel) + ["Predicted"] * len(time_sel),
                }
            )


            chart = (
                alt.Chart(plot_df)
                .mark_line()
                .encode(
                    x="DATE_TIME:T",
                    y="Value:Q",
                    color="Type:N",
                    tooltip=["DATE_TIME:T", "Type:N", "Value:Q"],
                )
                .properties(height=400, title=f"{target_choice} forecast – {plant_choice}")
            )
            st.altair_chart(chart, width='stretch')
            st.caption("Lines show actual vs predicted values over the selected time range.")



        # --- Overall metrics for each LSTM model on its test set ---
        st.markdown("### 3. Overall LSTM model performance (test sets)")

        metric_rows = []

        key_to_data_func = {
            "AC Power - Plant 1":   get_lstm_ac_power_plant1_data,
            "AC Power - Plant 2":   get_lstm_ac_power_plant2_data,
            "Yield - Plant 1":      get_lstm_yield_plant1_data,
            "Yield - Plant 2":      get_lstm_yield_plant2_data,
            "DC Power - Plant 1":   get_lstm_dc_power_plant1_data,
            "DC Power - Plant 2":   get_lstm_dc_power_plant2_data,
        }


        for key, model in lstm_models.items():
            if key not in key_to_data_func:
                continue

            data_k = key_to_data_func[key]()
            Xk = data_k["X_test"]
            y_lstm_test_scaled_k = data_k["y_test"]
            scaler_k = data_k["target_scaler"]

            try:
                # Predict in *scaled* space
                y_lstm_pred_scaled_k = lstm_predict(model, Xk)

                # Use the shared helper to inverse-transform + compute metrics
                mae, rmse, r2, _, _ = compute_lstm_metrics_from_scaled(
                    y_lstm_test_scaled_k,
                    y_lstm_pred_scaled_k,
                    scaler_k,
                )

                metric_rows.append(
                    {
                        "Model": key,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R²": r2,
                    }
                )

            except Exception as e:
                metric_rows.append(
                    {"Model": key, "MAE": np.nan, "RMSE": np.nan, "R²": np.nan}
                )
                st.warning(f"Could not compute metrics for '{key}': {e}")

        if metric_rows:
            metrics_df = pd.DataFrame(metric_rows).set_index("Model")
            st.dataframe(
                metrics_df.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.3f}"})
            )
        else:
            st.info("No LSTM metrics available.")



# ----- TAB 4: CLASSIFICATION MODELS -----
with tab_class:
    st.subheader("Operating condition classification")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### 1. Choose model and set inputs")

        model_choice = st.radio(
            "Classification model",
            options=["Logistic Regression", "Support Vector Machine", "Compare both"],
        )

        # Use dataset ranges to set sensible slider limits
        num_df = df.select_dtypes(include=["number"])
        dc_power = st.slider(
            "DC_POWER",
            float(num_df["DC_POWER"].min()),
            float(num_df["DC_POWER"].max()),
            float(num_df["DC_POWER"].median()),
        )
        ac_power = st.slider(
            "AC_POWER",
            float(num_df["AC_POWER"].min()),
            float(num_df["AC_POWER"].max()),
            float(num_df["AC_POWER"].median()),
        )
        amb_temp = st.slider(
            "AMBIENT_TEMPERATURE (°C)",
            float(num_df["AMBIENT_TEMPERATURE"].min()),
            float(num_df["AMBIENT_TEMPERATURE"].max()),
            float(num_df["AMBIENT_TEMPERATURE"].median()),
        )
        mod_temp = st.slider(
            "MODULE_TEMPERATURE (°C)",
            float(num_df["MODULE_TEMPERATURE"].min()),
            float(num_df["MODULE_TEMPERATURE"].max()),
            float(num_df["MODULE_TEMPERATURE"].median()),
        )
        irradiation = st.slider(
            "IRRADIATION",
            float(num_df["IRRADIATION"].min()),
            float(num_df["IRRADIATION"].max()),
            float(num_df["IRRADIATION"].median()),
        )

        plant_label = st.selectbox(
            "Plant",
            plant_options,
            key="class_plant_select",
        )

        input_df = build_input_row(
            dc_power,
            ac_power,
            amb_temp,
            mod_temp,
            irradiation,
        )

    with right_col:
        st.markdown("### 2. Model predictions")

        # Map numeric labels to readable names
        label_map = {
            0: "Suboptimal",
            1: "Optimal",
            "0": "Suboptimal",
            "1": "Optimal",
        }

        def show_single_model_results(name, model):
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                pred_idx = int(np.argmax(proba))
                raw_label = model.classes_[pred_idx]
            else:
                raw_label = model.predict(input_df)[0]

            # Pretty label for display
            pred_label_str = label_map.get(raw_label, str(raw_label))
            st.write(f"**{name} prediction:** {pred_label_str}")

            if proba is not None:
                st.write("Class probabilities:")
                class_names = [label_map.get(c, str(c)) for c in model.classes_]
                prob_df = pd.DataFrame(
                    {"class_name": class_names, "probability": proba}
                )
                st.bar_chart(
                    prob_df.set_index("class_name"),
                    width='stretch',
                )


        # --- Show per-input predictions (with bar charts) ---
        if model_choice == "Compare both":
            # two columns: Logistic Regression (left), SVM (right)
            col_log, col_svm = st.columns(2)     

            with col_log:
                st.markdown("#### Logistic Regression")
                show_single_model_results(
                    "Logistic Regression",
                    class_models["Logistic Regression"],
                )

            with col_svm:
                st.markdown("#### Support Vector Machine")
                show_single_model_results(
                    "Support Vector Machine",
                    class_models["Support Vector Machine"],
                )

        else:
            model = class_models[model_choice]
            show_single_model_results(model_choice, model)


        st.markdown("### 3. Overall performance on dataset")

        # --- Metrics on Scenario B test set ---
        if model_choice == "Compare both":
            eval_models = class_models
        else:
            eval_models = {model_choice: class_models[model_choice]}

        for name, model in eval_models.items():
            st.write(
                f"**{name} – metrics on Scenario B test set (n={len(y_clf_test)})**"
            )

            try:
                y_pred_test = model.predict(X_clf_test)
                y_true_test = y_clf_test

                if len(y_pred_test) != len(y_true_test):
                    st.warning(
                        f"{name}: predicted {len(y_pred_test)} samples for "
                        f"{len(y_true_test)} true labels; truncating to match for metrics."
                    )
                    min_len = min(len(y_pred_test), len(y_true_test))
                    y_pred_test = np.asarray(y_pred_test)[:min_len]
                    y_true_test = np.asarray(y_true_test)[:min_len]
                else:
                    y_true_test = np.asarray(y_true_test)

                report_dict = classification_report(
                    y_true_test, y_pred_test, output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))

                cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
                fig = plot_confusion_matrix(
                    cm, class_names=["Suboptimal (0)", "Optimal (1)"]
                )
                st.pyplot(fig)
                st.write(f"Total samples in confusion matrix: {cm.sum()}")

            except Exception as e:
                st.error(f"Error computing metrics for {name}: {e}")
