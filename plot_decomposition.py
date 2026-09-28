import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

from config import PROCESSED_DIR, TICKER


def decompose_ticker(ticker: str = TICKER, dataset: str = "default") -> None:
    parquet = PROCESSED_DIR / f"full_{dataset}.parquet"
    df = (
        pd.read_parquet(parquet)[["Date", "Ticker", "Close"]]
        .query("Ticker == @ticker")
        .set_index("Date")
        .sort_index()
    )
    df.index = pd.to_datetime(df.index)

    if df.empty:
        print(f"Ticker '{ticker}' not found in {parquet}.")
        return

    # Log-STL
    log_close = np.log(df["Close"])
    stl = STL(log_close, period=270, robust=True).fit()

    log_cycle, log_trend = sm.tsa.filters.hpfilter(
        stl.trend,
        lamb=129_600,  # Daily lambda
    )

    df_res = pd.DataFrame(index=df.index)
    df_res["Close"] = df["Close"]
    df_res["Trend"] = np.exp(log_trend)
    df_res["Seasonal"] = np.exp(stl.seasonal)
    df_res["Cyclical"] = np.exp(log_cycle)
    df_res["Residual"] = np.exp(stl.resid)

    # Plot
    titles = (
        "Close Price ($)",
        "Trend ($)",
        "Seasonal Factor (multiplier)",
        "Cyclical Factor (multiplier)",
        "Residual (noise)",
    )
    colours = ("black", "steelblue", "seagreen", "darkorange", "grey")

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=titles
    )

    for i, (col, colour) in enumerate(zip(df_res.columns, colours), start=1):
        fig.add_trace(
            go.Scatter(
                x=df_res.index,
                y=df_res[col],
                name=col,
                line=dict(color=colour, width=1.5),
            ),
            row=i,
            col=1,
        )
        if i > 2:
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="lightgrey",
                row=i,
                col=1,
            )

    fig.update_layout(
        height=1200,
        title_text=f"<b>{ticker} Multiplicative Decomposition (Log-STL + HP Filter)</b>",
        showlegend=False,
        template="plotly_white",
    )
    fig.show()

if __name__ == "__main__":
    decompose_ticker()