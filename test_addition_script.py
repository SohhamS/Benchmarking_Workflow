import os
import pandas as pd
from script import (
    HISTORICAL_CSV_PATH,
    PLOTS_DIR,
    MODELS_DIR,
    MODELS_FILE,
    load_and_preprocess_data,
    train_models_and_get_thresholds,
    plot_event_times,
    save_models,
    load_models,
)
from sklearn.linear_model import QuantileRegressor
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Base dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NEW_INPUTS_DIR = os.path.join(BASE_DIR, 'new_inputs')
RESULTS_DIR = os.path.join(BASE_DIR, 'historical_data', 'new_input_with_results')
COMPLETED_DIR = os.path.join(NEW_INPUTS_DIR, 'completed')


def ensure_dirs():
    os.makedirs(NEW_INPUTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(COMPLETED_DIR, exist_ok=True)


def classify_status(event_models, bundle_size, event_name, actual_time_seconds):
    info = event_models.get(event_name)
    if info is None:
        return 'unknown'  # No model for this event

    # Prefer quantile models when available; else fall back to OLS thresholds
    x_df = pd.DataFrame([[bundle_size]], columns=['log_bundle_size'])

    p50 = None
    p90 = None
    if info.get('q50_model') is not None:
        p50 = float(info['q50_model'].predict(x_df)[0])
    if info.get('q90_model') is not None:
        p90 = float(info['q90_model'].predict(x_df)[0])

    # If quantiles missing, derive threshold from OLS + residual_std
    if p50 is None:
        p50 = float(info['ols_model'].predict(x_df)[0])
    if p90 is None:
        p90 = float(info['ols_model'].predict(x_df)[0] + 2 * info.get('residual_std', 0.0))

    t = float(actual_time_seconds)
    if t <= p50:
        return 'expected'
    if t <= p90:
        return 'slow'
    return 'extremely slow'


def process_new_file(csv_path: str, historical_df: pd.DataFrame, event_models: dict) -> pd.DataFrame:
    logger.info(f"Processing new input: {csv_path}")
    df_new_raw = pd.read_csv(csv_path)
    # Reuse conversion from main script
    from script import time_to_seconds
    df_new = df_new_raw.copy()
    df_new['total_time_taken_seconds'] = df_new['total_time_taken'].apply(time_to_seconds)

    # Classify row-wise
    statuses = []
    for _, row in df_new.iterrows():
        event = row.get('event_name')
        size = row.get('log_bundle_size')
        time_sec = row.get('total_time_taken_seconds')
        if pd.isna(event) or pd.isna(size) or pd.isna(time_sec):
            statuses.append('unknown')
            continue
        statuses.append(classify_status(event_models, float(size), str(event), float(time_sec)))
    df_new['status'] = statuses

    # Save result file with same name into results dir (no renaming)
    base = os.path.basename(csv_path)
    out_path = os.path.join(RESULTS_DIR, base)
    df_new.to_csv(out_path, index=False)
    logger.info(f"Saved classified file -> {out_path}")

    # Append valid rows to historical_df (ensure required columns exist)
    # Prefer including log_bundle_id if present
    has_id = 'log_bundle_id' in df_new.columns
    appended_cols = ['log_bundle_id', 'log_bundle_size', 'total_time_taken', 'event_name'] if has_id else ['event_name', 'log_bundle_size', 'total_time_taken']
    missing = [c for c in appended_cols if c not in df_new.columns]
    if missing:
        logger.warning(f"Skipping append to historical: missing columns {missing}")
    else:
        # Keep original string time; historical script will recompute seconds
        # Reorder columns to match historical when possible
        if set(['log_bundle_id','log_bundle_size','total_time_taken','event_name']).issubset(historical_df.columns):
            order = ['log_bundle_id','log_bundle_size','total_time_taken','event_name']
            to_append = df_new[[c for c in order if c in df_new.columns]]
        else:
            to_append = df_new[appended_cols]
        historical_df = pd.concat([historical_df, to_append], ignore_index=True)
        logger.info(f"Appended {len(df_new)} rows to historical dataframe (in-memory)")

    return historical_df


def main():
    ensure_dirs()

    # Load historical data
    hist_df = pd.read_csv(HISTORICAL_CSV_PATH)

    # Load or train models from historical data
    models = load_models()
    if models is None:
        logger.info("No saved models; training from historical data...")
        df_proc = load_and_preprocess_data(HISTORICAL_CSV_PATH)
        models = train_models_and_get_thresholds(df_proc)
        save_models(models)

    # Process all new input CSVs
    files = [f for f in os.listdir(NEW_INPUTS_DIR) if f.lower().endswith('.csv')]
    if not files:
        logger.info("No new CSV files found in 'new_inputs/'. Nothing to do.")
        return

    for fname in files:
        csv_path = os.path.join(NEW_INPUTS_DIR, fname)
        hist_df = process_new_file(csv_path, hist_df, models)
        # Move processed file to completed dir
        dest = os.path.join(COMPLETED_DIR, fname)
        try:
            shutil.move(csv_path, dest)
            logger.info(f"Moved processed input -> {dest}")
        except Exception as e:
            logger.error(f"Failed to move {csv_path} to {dest}: {e}")

    # Persist updated historical CSV
    before = len(hist_df)
    # If log_bundle_id exists for most rows, dedupe on it first; else use composite key
    if 'log_bundle_id' in hist_df.columns:
        # Treat NaN IDs carefully: duplicate logic for NaN rows uses composite key
        with_id = hist_df.dropna(subset=['log_bundle_id'])
        without_id = hist_df[hist_df['log_bundle_id'].isna()] if hist_df['log_bundle_id'].isna().any() else hist_df.iloc[0:0]
        with_id = with_id.drop_duplicates(subset=['log_bundle_id','event_name','log_bundle_size','total_time_taken'])
        if len(without_id) > 0:
            without_id = without_id.drop_duplicates(subset=['event_name','log_bundle_size','total_time_taken'])
            hist_df = pd.concat([with_id, without_id], ignore_index=True)
        else:
            hist_df = with_id
    else:
        hist_df = hist_df.drop_duplicates(subset=['event_name', 'log_bundle_size', 'total_time_taken'])
    after = len(hist_df)
    removed = before - after
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows before saving historical data.")
    hist_df.to_csv(HISTORICAL_CSV_PATH, index=False)
    logger.info(f"Updated historical CSV -> {HISTORICAL_CSV_PATH}")

    # Retrain on updated historical, save models, and regenerate plots
    df_updated_proc = load_and_preprocess_data(HISTORICAL_CSV_PATH)
    models_updated = train_models_and_get_thresholds(df_updated_proc)
    save_models(models_updated)
    plot_event_times(df_updated_proc, models_updated, output_dir=PLOTS_DIR)
    logger.info("Retrained models and regenerated plots.")


if __name__ == '__main__':
    main() 