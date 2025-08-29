## Overview
This repository analyzes log bundle processing times per event, fits regression models per event to estimate expected time vs. bundle size, and generates plots to visualize trends and thresholds. It also classifies new input data as expected/slow/extremely slow, appends the results to historical data (with de-duplication), persists trained models, and regenerates plots.

## Folder structure
- `historical_data/`
  - `bundle_times.csv`: Master historical dataset used for training and plotting
  - `new_input_with_results/`: Classified copies of newly processed CSVs (same filename, with a `status` column)
  - `bundle_times_main_history.csv`: A copy/backup of a main history snapshot (optional)
- `new_inputs/`
  - Drop new CSVs here to classify
  - `completed/`: Processed files are moved here automatically
- `event_plots/`: Generated per-event PNG plots (overwritten on each run)
- `trained_models/`
  - `models.pkl`: Persisted per-event models (OLS + quantile regressors)
- `script.py`: Core training, plotting, and model save/load utilities
- `test_addition_script.py`: Batch classification of new inputs and historical updates

## Data schema
Expected CSV columns:
- `log_bundle_id` (optional but preferred): Identifier for a bundle run
- `log_bundle_size` (numeric): Bundle size used as the predictor (x-axis)
- `total_time_taken` (string): Time duration in `HH:MM:SS` (also supports `MM:SS`/`SS`)
- `event_name` (string): Event label (e.g., Parsing, Caching)


## Methodology
Per event, models are fitted to relate time to bundle size:
- OLS baseline: `LinearRegression` (thin red line) for context
- Quantile regression:
  - p50 (green): Median time vs. size, used as expected trend
  - p90 (orange dashed): High-percentile threshold for warnings
- Classification logic for a new row (size S, time T):
  - T ≤ p50(S) → expected
  - p50(S) < T ≤ p90(S) → slow
  - T > p90(S) → extremely slow
- If quantile models are unavailable for an event, fallback threshold is `OLS(S) + 2·residual_std`.

## Plots
- One PNG per event saved in `event_plots/`:
  - Scatter: historical points (seconds vs. bundle size)
  - Lines: OLS (baseline), p50 (median), p90 (threshold)
- Plots overwrite on each run and use a non-interactive backend.

## How to classify a new run

Given a new bundle with:
size = S (x-axis), actual time = T (y-axis)

Read the y-values from the lines at x = S:
p50_S = p50(S)
p90_S = p90(S)

Then:
• OK (fast/expected): T ≤ p50_S
• OK (slower but normal): p50_S < T ≤ p90_S
• WARNING (slow/outlier): T > p90_S


### Process new input CSVs
1. Place new CSV files into `new_inputs/`
2. Run:
```bash
python3 /home/nutanix/Desktop/sohham/benchmarking_data/test_addition_script.py
```
3. What happens:
   - Each file is loaded and a `status` column is added per row (expected/slow/extremely slow)
   - A classified copy is saved to `historical_data/new_input_with_results/` (same filename)
   - Rows are appended to `historical_data/bundle_times.csv` with de-duplication
   - Models are retrained on the updated historical data and saved to `trained_models/models.pkl`
   - Plots are regenerated in `event_plots/`
   - Processed input files are moved to `new_inputs/completed/`

## De-duplication rules when merging
- If `log_bundle_id` is present: de-duplicate on `log_bundle_id,event_name,log_bundle_size,total_time_taken`
- Rows without `log_bundle_id`: de-duplicate on `event_name,log_bundle_size,total_time_taken`
- A log line reports the number of duplicates removed