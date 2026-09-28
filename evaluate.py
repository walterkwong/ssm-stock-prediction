import glob
from pathlib import Path
import numpy as np
import polars as pl

from config import RESULTS_DIR

OUTPUT_FILE = RESULTS_DIR / "all_models_evaluation.csv"


def rank_ic(df: pl.DataFrame) -> float:
    values = []
    # Group by date
    for group in df.partition_by("Date", as_dict=False):
        if group.height < 2:
            continue
        pred = group["Predicted_Return"].to_numpy()
        actual = group["Actual_Return"].to_numpy()
        pred_rank = np.argsort(np.argsort(pred)).astype(float)
        actual_rank = np.argsort(np.argsort(actual)).astype(float)
        values.append(np.corrcoef(pred_rank, actual_rank)[0, 1])
    return float(np.nanmean(values)) if values else 0.0


def ls_spread(df: pl.DataFrame) -> float:
    spreads = []
    for group in df.partition_by("Date", as_dict=False):
        if group.height < 2:
            continue

        k = max(1, int(group.height * 0.10))
        order = np.argsort(group["Predicted_Return"].to_numpy())
        actual = group["Actual_Return"].to_numpy()
        spreads.append(actual[order[-k:]].mean() - actual[order[:k]].mean())

    return float(np.mean(spreads)) if spreads else 0.0


def sharpe(df: pl.DataFrame, annualize: bool = True) -> float:
    # Daily portfolio returns
    daily_returns = (
        df.with_columns(
            (pl.col("Predicted_Return").sign() * pl.col("Actual_Return")).alias("pos_return")
        )
        .group_by("Date")
        .agg(pl.col("pos_return").mean().alias("daily_return"))
        .sort("Date")
        ["daily_return"]
        .to_numpy()
    )

    std = daily_returns.std()
    if len(daily_returns) == 0 or std == 0:
        return 0.0

    factor = np.sqrt(252) if annualize else 1.0
    return float((daily_returns.mean() / std) * factor)


def calculate_metrics(filepath: Path | str) -> dict:
    df = pl.read_csv(filepath)

    predictions = df["Predicted_Return"].to_numpy()
    actuals = df["Actual_Return"].to_numpy()

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    directional_accuracy = float(
        (np.sign(predictions) == np.sign(actuals)).mean()
    )

    return {
        "RMSE": rmse,
        "Directional_Accuracy": directional_accuracy,
        "Rank_IC": rank_ic(df),
        "LS_Spread": ls_spread(df),
        "Sharpe": sharpe(df),
    }


def evaluate_all(results_dir: Path = RESULTS_DIR):
    pattern = str(results_dir / "*_final_predictions.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No prediction files found matching: {pattern}")
        return None

    print(f"Found {len(files)} prediction files ...\n")

    rows = []
    for filepath in files:
        model_name = Path(filepath).stem.replace("_final_predictions", "")
        print(f"  Evaluating: {model_name}")
        metrics = calculate_metrics(filepath)
        metrics["Model"] = model_name
        rows.append(metrics)

    summary = pl.DataFrame(rows).select([
        "Model",
        "RMSE",
        "Directional_Accuracy",
        "Rank_IC",
        "LS_Spread",
        "Sharpe",
    ])
    summary.write_csv(OUTPUT_FILE)

    print(f"\nSaved evaluation summary -> {OUTPUT_FILE}")
    print(summary)
    return summary


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate_all()