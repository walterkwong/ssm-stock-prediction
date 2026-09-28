from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PROCESSED_DIR, RESULTS_DIR, TICKER

_PALETTE = ["blue", "green", "red", "orange", "purple", "brown"]


def _merge_predictions_with_close(
    predictions: pd.DataFrame,
    processed: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:
    pred_t = predictions[predictions["Ticker"] == ticker].copy()
    proc_t = processed[processed["Ticker"] == ticker][["Date", "Close"]].copy()

    merged = pred_t.merge(proc_t, on="Date")
    merged["Predicted_Close"] = merged["Close"] * (1 + merged["Predicted_Return"])
    merged["Actual_Close"] = merged["Close"] * (1 + merged["Actual_Return"])
    return merged


def _load_processed_close(dataset: str = "default") -> pd.DataFrame:
    path = PROCESSED_DIR / f"full_{dataset}.parquet"
    df = pd.read_parquet(path)[["Date", "Ticker", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_single_ticker(
    model_file: str | Path,
    processed: pd.DataFrame | None = None,
) -> None:
    model_file = Path(model_file)
    df = pd.read_csv(model_file)
    df["Date"] = pd.to_datetime(df["Date"])
    processed = processed if processed is not None else _load_processed_close()

    tickers = sorted(df["Ticker"].unique())
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        merged = _merge_predictions_with_close(df, processed, ticker)
        visible = i == 0

        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged["Predicted_Close"],
                mode="lines",
                name="Predicted",
                line=dict(color="blue"),
                visible=visible,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged["Actual_Close"],
                mode="lines",
                name="Actual",
                line=dict(color="red"),
                visible=visible,
            )
        )

    buttons = []
    for i, ticker in enumerate(tickers):
        visible = [False] * len(fig.data)
        visible[i * 2] = True
        visible[i * 2 + 1] = True

        buttons.append(
            dict(
                label=ticker,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"{ticker} Predicted vs Actual Close"},
                ],
            )
        )

    fig.update_layout(
        title=f"{tickers[0]} Predicted vs Actual Close",
        xaxis_title="Date",
        yaxis_title="Close Price ($)",
        hovermode="x unified",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    fig.show()


def plot_ticker_grid(
    tickers: list[str],
    n_cols: int = 4,
    processed: pd.DataFrame | None = None,
    results_dir: Path = RESULTS_DIR,
) -> None:
    pred_files = sorted(results_dir.glob("*_final_predictions.csv"))
    if not pred_files:
        print(f"No prediction files found in '{results_dir}'.")
        return

    model_names = [
        f.stem.replace("_final_predictions", "")
        for f in pred_files
    ]
    processed = processed if processed is not None else _load_processed_close()

    n_rows = (len(tickers) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=tickers,
    )

    predictions = []
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        df["Date"] = pd.to_datetime(df["Date"])
        predictions.append(df)

    for idx, ticker in enumerate(tickers):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Actual
        merged = _merge_predictions_with_close(
            predictions[0],
            processed,
            ticker,
        )
        fig.add_trace(
            go.Scatter(
                x=merged["Date"],
                y=merged["Actual_Close"],
                mode="lines",
                name="Actual",
                line=dict(color="black", width=2),
                showlegend=(idx == 0),
                hovertemplate="Actual: $%{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Predictions
        for m_idx, (df, m_name) in enumerate(
            zip(predictions, model_names)
        ):
            merged = _merge_predictions_with_close(df, processed, ticker)

            fig.add_trace(
                go.Scatter(
                    x=merged["Date"],
                    y=merged["Predicted_Close"],
                    mode="lines",
                    name=m_name,
                    line=dict(color=_PALETTE[m_idx % len(_PALETTE)]),
                    showlegend=(idx == 0),
                    hovertemplate=f"{m_name}: $"
                    + "%{y:.2f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=max(600, n_rows * 250),
        showlegend=True,
        hovermode="x unified",
    )
    fig.show()


if __name__ == "__main__":
    pred_files = sorted(RESULTS_DIR.glob("*_final_predictions.csv"))

    if not pred_files:
        print("No prediction files found. Train models first.")
        raise SystemExit(1)

    processed = _load_processed_close()

    plot_single_ticker(
        pred_files[0],
        processed=processed,
    )