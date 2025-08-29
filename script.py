import os
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from datetime import datetime

# Base directory for absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (absolute)
HISTORICAL_CSV_PATH = os.path.join(BASE_DIR, 'historical_data', 'bundle_times.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'event_plots')
MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
MODELS_FILE = os.path.join(MODELS_DIR, 'models.pkl')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def time_to_seconds(time_str):
    if pd.isnull(time_str):
        return None
    parts = str(time_str).split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2: # Handle MM:SS format if it exists
        m, s = map(int, parts)
        return m * 60 + s
    elif len(parts) == 1: # Handle SS format if it exists
        return int(parts[0])
    return None


def load_and_preprocess_data(file_path):
    logger.info(f"Loading CSV from: {file_path}")
    df = pd.read_csv(file_path)
    before_rows = len(df)
    df['total_time_taken_seconds'] = df['total_time_taken'].apply(time_to_seconds)
    after_rows = len(df.dropna(subset=['log_bundle_size', 'total_time_taken_seconds']))
    logger.info(f"Loaded {before_rows} rows; {after_rows} rows with valid size and time.")
    return df


def train_models_and_get_thresholds(df):
    logger.info("Training regression models per event and computing thresholds...")
    event_threshold_info = {}

    for event_name in df['event_name'].unique():
        event_df = df[df['event_name'] == event_name].copy()
        event_df = event_df.dropna(subset=['log_bundle_size', 'total_time_taken_seconds'])
        n = len(event_df)
        if n < 2:
            logger.warning(f"Skipping '{event_name}': not enough data points ({n}).")
            continue
        
        X = event_df[['log_bundle_size']]
        y = event_df['total_time_taken_seconds']

        # OLS baseline
        ols = LinearRegression()
        ols.fit(X, y)
        ols_pred = ols.predict(X)
        ols_resid = y - ols_pred
        residual_std = float(np.std(ols_resid)) if len(ols_resid) > 0 else 0.0

        # Quantile regressions: p50 (median) and p90 (high threshold)
        try:
            q50 = QuantileRegressor(quantile=0.5, alpha=1e-4)
            q50.fit(X, y)
            q90 = QuantileRegressor(quantile=0.9, alpha=1e-4)
            q90.fit(X, y)
        except Exception as e:
            logger.error(f"Quantile regression failed for '{event_name}': {e}")
            q50 = None
            q90 = None

        event_threshold_info[event_name] = {
            'ols_model': ols,
            'residual_std': residual_std,
            'upper_bound_multiplier': 2,
            'q50_model': q50,
            'q90_model': q90,
        }

        logger.info(
            f"Trained '{event_name}': n={n}, OLS coef={ols.coef_[0]:.6f}, intercept={ols.intercept_:.2f}, resid_std={residual_std:.2f}"
        )
        if q50 is not None:
            logger.info(f"  q50 coef={float(q50.coef_[0]):.6f}, intercept={float(q50.intercept_):.2f}")
        if q90 is not None:
            logger.info(f"  q90 coef={float(q90.coef_[0]):.6f}, intercept={float(q90.intercept_):.2f}")

    logger.info(f"Finished training {len(event_threshold_info)} event models.")
    return event_threshold_info


def save_models(event_threshold_info, model_dir: str = MODELS_DIR, models_file: str = MODELS_FILE):
    os.makedirs(model_dir, exist_ok=True)
    payload = {
        'event_threshold_info': event_threshold_info,
        'metadata': {
            'saved_at': datetime.utcnow().isoformat() + 'Z'
        }
    }
    joblib.dump(payload, models_file)
    logger.info(f"Saved models to {models_file}")


def load_models(model_dir: str = MODELS_DIR, models_file: str = MODELS_FILE):
    if not os.path.exists(models_file):
        logger.info(f"Models file not found at {models_file}")
        return None
    payload = joblib.load(models_file)
    logger.info(f"Loaded models from {models_file}")
    return payload.get('event_threshold_info', None)


def predict_and_check_bundle(event_threshold_info, new_bundle_size, event_name, actual_time_taken_seconds):
    if event_name not in event_threshold_info:
        return f"No model found for event: {event_name}. Cannot predict or check threshold."

    info = event_threshold_info[event_name]
    model = info['ols_model']
    residual_std = info['residual_std']
    upper_bound_multiplier = info['upper_bound_multiplier']

    prediction_input = pd.DataFrame([[new_bundle_size]], columns=['log_bundle_size'])
    predicted_time = model.predict(prediction_input)[0]
    
    upper_threshold = predicted_time + (upper_bound_multiplier * residual_std)
    
    status = "OK"
    warning_message = ""
    if actual_time_taken_seconds > upper_threshold:
        status = "WARNING"
        warning_message = (f"Actual time ({actual_time_taken_seconds:.2f}s) exceeds upper threshold "
                           f"({upper_threshold:.2f}s) for bundle size {new_bundle_size}.")

    return {
        'event_name': event_name,
        'bundle_size': new_bundle_size,
        'predicted_time': predicted_time,
        'upper_threshold': upper_threshold,
        'actual_time_taken': actual_time_taken_seconds,
        'status': status,
        'warning_message': warning_message
    }


def _sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in name.strip())
    return safe or "event"


def plot_event_times(df, event_threshold_info, output_dir: str = PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving per-event plots to directory: {output_dir}")

    unique_events = df['event_name'].unique()
    for event_name in unique_events:
        event_df = df[df['event_name'] == event_name].copy()
        event_df = event_df.dropna(subset=['log_bundle_size', 'total_time_taken_seconds'])
        n = len(event_df)
        if n == 0:
            logger.warning(f"Skipping plot for '{event_name}': no valid rows.")
            continue

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='log_bundle_size', y='total_time_taken_seconds', data=event_df, label='Actual Time')

        x_min = event_df['log_bundle_size'].min()
        x_max = event_df['log_bundle_size'].max()
        if pd.isna(x_min) or pd.isna(x_max):
            logger.warning(f"Skipping overlays for '{event_name}': invalid x-range.")
        else:
            x_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)
            x_range_df = pd.DataFrame(x_range, columns=['log_bundle_size'])

            info = event_threshold_info.get(event_name)
            if info is not None:
                # Plot OLS line for reference (thin)
                ols = info['ols_model']
                y_pred_ols = ols.predict(x_range_df)
                sns.lineplot(x=x_range.flatten(), y=y_pred_ols, color='red', alpha=0.5, linewidth=1.2, label='OLS (baseline)')

                # Plot quantile lines if available
                if info['q50_model'] is not None:
                    y_q50 = info['q50_model'].predict(x_range_df)
                    sns.lineplot(x=x_range.flatten(), y=y_q50, color='green', label='p50 (median)')
                if info['q90_model'] is not None:
                    y_q90 = info['q90_model'].predict(x_range_df)
                    sns.lineplot(x=x_range.flatten(), y=y_q90, color='orange', linestyle='--', label='p90 (threshold)')

        plt.title(f"{event_name}")
        plt.xlabel('Log Bundle Size')
        plt.ylabel('Time Taken (seconds)')
        plt.grid(True)
        legend = plt.legend(title=f"Event: {event_name}")
        if legend is not None:
            legend._legend_box.align = "left"

        filename = _sanitize_filename(event_name) + ".png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"Saved plot for '{event_name}' -> {filepath}")


if __name__ == "__main__":
    file_path = HISTORICAL_CSV_PATH
    df = load_and_preprocess_data(file_path)

    logger.info("Trying to load existing models...")
    models = load_models()
    if models is None:
        logger.info("No existing models found. Training from historical data...")
        models = train_models_and_get_thresholds(df)
        save_models(models)
    else:
        logger.info("Using loaded models to plot.")

    logger.info("Generating and saving plots (with p50/p90)...")
    plot_event_times(df, models)

    logger.info("Done.")
