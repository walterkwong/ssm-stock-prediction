import polars as pl
import numpy as np
import pywt
import glob
import ast
from pathlib import Path
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import polars.selectors as cs

from config import DATA_DIR, GAP_DAYS, PROCESSED_DIR, FINAL_WARMUP, WAVELET_WINDOW, TDA_WINDOW, STRIDE

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_VOL_COLS = ["Volume", "log_vol_change", "log_vol_change_n"]


def load_and_clean_universe(data_dir) -> pl.DataFrame:
    all_files = glob.glob(f"{data_dir}/*.parquet")
    ticker_dfs = []
    print(f"Found {len(all_files)} files in {data_dir}")

    for file_path in all_files:
        try:
            df = pl.read_parquet(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        rename_map = {}
        current_ticker = None

        # handle multi-index columns
        for col in df.columns:
            if col == "Date":
                continue
            if col.startswith("('"):
                parsed_tuple = ast.literal_eval(col)
                rename_map[col] = parsed_tuple[0]
                current_ticker = parsed_tuple[1]
            else:
                rename_map[col] = col
                if current_ticker is None:
                    current_ticker = Path(file_path).stem

        if not rename_map:
            continue

        df = df.rename(rename_map)
        if "Ticker" not in df.columns:
            if current_ticker:
                df = df.with_columns(pl.lit(current_ticker).alias("Ticker"))
            else:
                continue

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in numeric_cols if c in df.columns]

        df = df.with_columns([
            pl.col("Date").cast(pl.Datetime),
            *[pl.col(c).cast(pl.Float64) for c in available_cols],
        ])
        df = df.select(["Date", "Ticker"] + available_cols)
        ticker_dfs.append(df)

    if not ticker_dfs:
        raise ValueError("No valid data found!")

    return pl.concat(ticker_dfs).sort(["Date", "Ticker"])


def extract_tda_features(data: np.ndarray):
    # Persistence entropy
    if len(data) < 5:
        return 0.0, 0.0
    try:
        X = data.reshape(1, -1, 1)
        diagrams = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(X)
        entropy = PersistenceEntropy().fit_transform(diagrams)
        return entropy[0][0], entropy[0][1]
    except Exception:
        return 0.0, 0.0


def wavelet_denoise(data: np.ndarray, wavelet="db4", level=1):
    # Split into trend and noise
    if len(data) < 2:
        return np.zeros_like(data), np.zeros_like(data)
    try:
        coeffs = pywt.wavedec(data, wavelet, mode="per", level=level)
        trend = pywt.upcoef("a", coeffs[0], wavelet, level=level, take=len(data))
        noise = pywt.upcoef("d", coeffs[1], wavelet, level=level, take=len(data))
        return trend, noise
    except Exception:
        return data, np.zeros_like(data)


def apply_rolling_wavelet(group):
    # Causal rolling window
    rets = group["log_return"].fill_null(0).to_numpy()
    n = len(rets)
    trend_arr = np.zeros(n)
    noise_arr = np.zeros(n)

    for i in range(WAVELET_WINDOW, n, STRIDE):
        segment = rets[i - WAVELET_WINDOW:i]
        seg_trend, seg_noise = wavelet_denoise(segment)
        fill_end = min(i + STRIDE, n)
        trend_arr[i:fill_end] = seg_trend[-1]
        noise_arr[i:fill_end] = seg_noise[-1]

    return group.with_columns([
        pl.Series("w_trend", trend_arr, dtype=pl.Float64),
        pl.Series("w_noise", noise_arr, dtype=pl.Float64),
    ])


def apply_rolling_tda(group):
    # Causal rolling window
    rets = group["log_return"].fill_null(0).to_numpy()
    n = len(rets)
    h0_arr = np.zeros(n)
    h1_arr = np.zeros(n)

    for i in range(TDA_WINDOW, n, STRIDE):
        h0, h1 = extract_tda_features(rets[i - TDA_WINDOW:i])
        fill_end = min(i + STRIDE, n)
        h0_arr[i:fill_end] = h0
        h1_arr[i:fill_end] = h1

    return group.with_columns([
        pl.Series("tda_h0", h0_arr, dtype=pl.Float64),
        pl.Series("tda_h1", h1_arr, dtype=pl.Float64),
    ])


def add_base_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        (pl.col("Close").log() - pl.col("Close").shift(1).over("Ticker").log()).alias("log_return"),
        (0.5 * (pl.col("High").log() - pl.col("Low").log()) ** 2
         - (2 * np.log(2) - 1) * (pl.col("Close").log() - pl.col("Open").log()) ** 2).alias("gk_vol"),
        (pl.col("Volume").log().diff().over("Ticker")).alias("log_vol_change"),
    ])


def add_market_relative(df: pl.DataFrame) -> pl.DataFrame:
    # Daily median as a simple market proxy
    mkt = df.group_by("Date").agg(pl.col("log_return").median().alias("mkt_ret"))
    return df.join(mkt, on="Date").with_columns(
        (pl.col("log_return") - pl.col("mkt_ret")).alias("rel_ret")
    )


def clean_volume_cols(df: pl.DataFrame) -> pl.DataFrame:
    # Remove invalid volume values
    df = df.with_columns([
        pl.col(c).replace(float("inf"), None).replace(float("-inf"), None).replace(float("nan"), None)
        for c in _VOL_COLS
    ])
    return df


def normalize_and_target(df: pl.DataFrame, cols_to_norm: list) -> pl.DataFrame:
    for c in cols_to_norm:
        roll_m = pl.col(c).rolling_mean(GAP_DAYS).over("Ticker")
        roll_s = pl.col(c).rolling_std(GAP_DAYS).over("Ticker")
        df = df.with_columns(
            ((pl.col(c) - roll_m) / (roll_s + 1e-8)).clip(-3, 3).alias(f"{c}_n")
        )

    df = df.with_columns(
        pl.col("log_return").shift(-1).over("Ticker").alias("target_next_ret")
    )
    return df.with_columns(cs.numeric().cast(pl.Float64))


# pipelines

def pipeline_default(df: pl.DataFrame) -> pl.DataFrame:
    print("Running default pipeline...")
    df = add_base_features(df)
    df = add_market_relative(df)
    cols = ["log_return", "gk_vol", "rel_ret", "log_vol_change"]
    df = normalize_and_target(df, cols)
    return clean_volume_cols(df)


def pipeline_wavelet(df: pl.DataFrame) -> pl.DataFrame:
    print("Running wavelet pipeline...")
    df = add_base_features(df)
    df = add_market_relative(df)
    print("  -> rolling wavelets...")
    df = df.group_by("Ticker").map_groups(apply_rolling_wavelet)
    cols = ["log_return", "gk_vol", "rel_ret", "log_vol_change", "w_trend", "w_noise"]
    df = normalize_and_target(df, cols)
    return clean_volume_cols(df)


def pipeline_tda(df: pl.DataFrame) -> pl.DataFrame:
    print("Running TDA pipeline...")
    df = add_base_features(df)
    df = add_market_relative(df)
    print("  -> rolling topology...")
    df = df.group_by("Ticker").map_groups(apply_rolling_tda)
    cols = ["log_return", "gk_vol", "rel_ret", "log_vol_change", "tda_h0", "tda_h1"]
    df = normalize_and_target(df, cols)
    return clean_volume_cols(df)


def pipeline_tda_wavelet(df: pl.DataFrame) -> pl.DataFrame:
    print("Running combined TDA + wavelet pipeline...")
    df = add_base_features(df)
    df = add_market_relative(df)
    print("  -> rolling topology...")
    df = df.group_by("Ticker").map_groups(apply_rolling_tda)
    print("  -> rolling wavelets...")
    df = df.group_by("Ticker").map_groups(apply_rolling_wavelet)
    cols = ["log_return", "gk_vol", "rel_ret", "log_vol_change",
            "tda_h0", "tda_h1", "w_trend", "w_noise"]
    df = normalize_and_target(df, cols)
    return clean_volume_cols(df)


def remove_bad_rows(df: pl.DataFrame) -> pl.DataFrame:
    feature_cols = [
        c for c in df.columns
        if c not in {"Date", "Ticker", "target_next_ret"}
    ]
    condition = pl.col("target_next_ret").is_not_null() & pl.col("target_next_ret").is_finite()

    for col in feature_cols:
        condition &= pl.col(col).is_not_null() & pl.col(col).is_finite()

    return df.filter(condition)


def synchronize_rows(df: pl.DataFrame, warmup_period: int) -> pl.DataFrame:
    # Remove warmup rows and invalid values
    df = df.with_columns(
        pl.int_range(0, pl.len()).over("Ticker").alias("row_nr")
    )
    df = df.filter(pl.col("row_nr") >= warmup_period).drop("row_nr")
    return remove_bad_rows(df)


def apply_and_save(df: pl.DataFrame, name: str, pipeline_func):
    df_eng = synchronize_rows(pipeline_func(df), FINAL_WARMUP)
    path = PROCESSED_DIR / f"full_{name}.parquet"
    df_eng.write_parquet(path)
    print(f"  -> saving {name}: {df_eng.shape} to {path}")


if __name__ == "__main__":
    print(f"Loading universe from {DATA_DIR}...")
    df_raw = load_and_clean_universe(DATA_DIR)
    print(f"Loaded universe: {df_raw.shape}")

    pipelines = {
        "default": pipeline_default,
        "wavelet": pipeline_wavelet,
        "tda": pipeline_tda,
        "tda_wavelet": pipeline_tda_wavelet,
    }

    for name, func in pipelines.items():
        apply_and_save(df_raw.clone(), name, func)
        print(f"Completed {name}.\n")