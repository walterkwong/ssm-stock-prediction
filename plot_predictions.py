import glob

import pandas as pd
import plotly.graph_objects as go

from config import RESULTS_DIR


def _load_prediction_files(results_dir=RESULTS_DIR) -> list[tuple[str, pd.DataFrame]]:
    pattern = str(results_dir / "*_final_predictions.csv")
    files = sorted(glob.glob(pattern))
    out = []

    for path in files:
        name = path.split("/")[-1].replace("_final_predictions.csv", "").upper()
        try:
            out.append((name, pd.read_csv(path)))
        except Exception as exc:
            print(f"  Skipping {path}: {exc}")

    return out


def plot_prediction_timeseries(results_dir=RESULTS_DIR) -> None:
    data = _load_prediction_files(results_dir)
    if not data:
        print(f"No prediction files found in '{results_dir}'.")
        return

    fig = go.Figure()

    _, df_first = data[0]
    fig.add_trace(
        go.Scatter(
            x=df_first.index,
            y=df_first["Actual_Return"],
            mode="lines",
            name="<b>ACTUAL RETURNS</b>",
            line=dict(color="black", width=2.5),
            opacity=0.45,
        )
    )

    for i, (name, df) in enumerate(data):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Predicted_Return"],
                mode="lines",
                name=name,
                line=dict(width=1.5),
                hovertemplate=f"<b>{name}</b><br>Index: %{{x}}<br>Pred: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="<b>Model Predictions vs Actual Returns</b>",
        xaxis_title="Time Index (Test Set)",
        yaxis_title="Log Return",
        legend_title="Models",
        hovermode="x unified",
        width=1100,
        height=600,
    )
    fig.show()


def plot_prediction_scatter(results_dir=RESULTS_DIR) -> None:
    data = _load_prediction_files(results_dir)
    if not data:
        return

    fig = go.Figure()
    all_vals = []

    for name, df in data:
        fig.add_trace(
            go.Scatter(
                x=df["Actual_Return"],
                y=df["Predicted_Return"],
                mode="markers",
                name=name,
                marker=dict(opacity=0.45, size=5),
            )
        )
        all_vals.extend(df["Actual_Return"].tolist())
        all_vals.extend(df["Predicted_Return"].tolist())

    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        fig.add_trace(
            go.Scatter(
                x=[mn, mx],
                y=[mn, mx],
                mode="lines",
                name="Perfect Fit",
                line=dict(color="black", dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="<b>Predicted vs Actual Returns</b>",
        xaxis_title="Actual Return",
        yaxis_title="Predicted Return",
        width=800,
        height=700,
    )
    fig.show()


if __name__ == "__main__":
    plot_prediction_timeseries()
    plot_prediction_scatter()