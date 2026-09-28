import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import RESULTS_DIR

_INPUT = RESULTS_DIR / "all_models_evaluation.csv"
_OUTPUT = "model_performance_comparison.png"


def plot_metrics(input_path=_INPUT, output_path=_OUTPUT) -> None:
    if not input_path.exists():
        print(f"Evaluation file not found: {input_path}")
        print("Run `python src/evaluation/metrics.py` first.")
        return

    df = pd.read_csv(input_path, on_bad_lines="skip")
    df["Model"] = df["Model"].str.lower()
    df = df[df["Model"].str.contains("benchmark|mamba")].sort_values("Model")

    metrics = ["Sharpe", "Rank_IC", "LS_Spread"]
    colours = ("steelblue", "seagreen", "darkorange")

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Sharpe", "Rank IC", "Long-Short Spread"),
        vertical_spacing=0.10,
    )

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=df["Model"],
                y=df[metric],
                marker_color=colours[i],
                name=metric,
                text=df[metric].apply(lambda v: f"{v:.4f}"),
                textposition="auto",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=1400,
        width=1200,
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=45)
    fig.write_image(str(output_path))
    print(f"Chart saved: {output_path}")
    fig.show()


if __name__ == "__main__":
    plot_metrics()