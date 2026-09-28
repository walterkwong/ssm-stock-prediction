from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import RESULTS_DIR

_EXCLUDE = {"final_results", "predictions", "evaluation", "benchmark_results"}


def plot_training_history(results_dir: Path = RESULTS_DIR) -> None:
    files = [
        f for f in sorted(results_dir.glob("*.csv"))
        if not any(exc in f.name for exc in _EXCLUDE)
    ]

    if not files:
        print(f"No epoch log files found in '{results_dir}'.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colours = ["blue", "red", "green", "purple", "orange", "cyan"]

    for i, path in enumerate(files):
        model_name = path.stem.upper()
        colour = colours[i % len(colours)]

        try:
            df = pd.read_csv(path)

            fig.add_trace(
                go.Scatter(
                    x=df["Epoch"],
                    y=df["TrainLoss"],
                    name=f"{model_name} Train",
                    line=dict(color=colour, width=2),
                    legendgroup=model_name,
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=df["Epoch"],
                    y=df["TestLoss"],
                    name=f"{model_name} Test",
                    line=dict(color=colour, width=2, dash="dot"),
                    legendgroup=model_name,
                ),
                secondary_y=False,
            )
        except Exception as exc:
            print(f"  Skipping {path}: {exc}")

    fig.update_layout(
        template="plotly_white",
        title="<b>Mamba Training Curves</b>",
        xaxis_title="Epoch",
        width=1100,
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_yaxes(title_text="<b>Loss</b>", secondary_y=False)
    fig.show()


if __name__ == "__main__":
    plot_training_history()