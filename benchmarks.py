# Cross-sectional neural network baselines

from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tsai.models.InceptionTime import InceptionTime
from tsai.models.RNN_FCN import LSTM_FCN
from tsai.models.TST import TST

from config import (
    BATCH_SIZE,
    BENCHMARK_LR,
    DATASETS,
    DEVICE,
    EPOCHS,
    GRAD_CLIP,
    PROCESSED_DIR,
    RESULTS_DIR,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from model import CrossSectionalDataset, collate_batch
from train import (
    TradingLoss,
    calculate_ls_spread,
    calculate_rank_ic,
    calculate_sharpe,
    clean_data,
    scale_features,
    set_seed,
    split_train_validation_test,
)


MODELS_TO_RUN = [
    "inception",
    "lstm_fcn",
    "tst",
]

RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"
EXCLUDE_COLS = {"Date", "Ticker", "target_next_ret"}


def make_model(name, input_channels, n_stocks):
    if name == "inception":
        model = InceptionTime(
            c_in=input_channels,
            c_out=n_stocks,
            nf=64,
            depth=9,
        )
    elif name == "lstm_fcn":
        model = LSTM_FCN(
            c_in=input_channels,
            c_out=n_stocks,
            seq_len=WINDOW_SIZE,
            hidden_size=256,
            rnn_layers=2,
        )
    elif name == "tst":
        model = TST(
            c_in=input_channels,
            c_out=n_stocks,
            seq_len=WINDOW_SIZE,
            d_model=256,
            n_layers=6,
            n_heads=16,
            d_ff=1024,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    return model


def flatten_cross_section(x):
    """Convert to tsai input format"""
    batch, time, stocks, features = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch, stocks * features, time)


def train_model(model, loader, epochs):
    criterion = TradingLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BENCHMARK_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=BENCHMARK_LR,
        total_steps=max(len(loader) * epochs, 1),
    )

    start = datetime.now()
    model.train()

    for epoch in range(epochs):
        for x, y, mask, *_ in loader:
            x = flatten_cross_section(x).to(DEVICE).float()
            y = y.to(DEVICE).float()
            mask = mask.to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y, mask)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

    return start, datetime.now()


def predict(model, loader, tickers):
    model.eval()
    predictions, actuals, masks, dates = [], [], [], []

    with torch.no_grad():
        for x, y, mask, batch_dates, _ in loader:
            preds = model(flatten_cross_section(x).to(DEVICE).float())
            predictions.append(preds.cpu().numpy())
            actuals.append(y.numpy())
            masks.append(mask.numpy())
            dates.extend(batch_dates)

    return (
        np.concatenate(predictions),
        np.concatenate(actuals),
        np.concatenate(masks),
        dates,
        tickers,
    )


def save_predictions(model_name, preds, actuals, masks, dates, tickers):
    rows = []
    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            if masks[i, j] == 0:
                continue
            rows.append((date, ticker, preds[i, j], actuals[i, j]))

    path = RESULTS_DIR / f"{model_name}_final_predictions.csv"
    pl.DataFrame(
        rows,
        schema=["Date", "Ticker", "Predicted_Return", "Actual_Return"],
        orient="row",
    ).write_csv(path)
    return path


def run_benchmarks():
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}\nDATASET: {dataset_name}\n{'=' * 60}")

        full_data = pl.read_parquet(
            PROCESSED_DIR / f"full_{dataset_name}.parquet"
        )
        full_data = clean_data(full_data)
        train_raw, val_raw, test_raw = split_train_validation_test(full_data)
        tickers = sorted(train_raw["Ticker"].unique().to_list())

        # Build final train + validation set
        final_train_raw = pl.concat([train_raw, val_raw], how="vertical")
        final_train_df, [final_test_df] = scale_features(
            final_train_raw,
            [test_raw],
        )

        final_train_ds = CrossSectionalDataset(
            final_train_df,
            final_train_df,
            WINDOW_SIZE,
            tickers,
        )
        final_test_ds = CrossSectionalDataset(
            final_train_df,
            final_test_df,
            WINDOW_SIZE,
            tickers,
        )
        final_train_loader = DataLoader(
            final_train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=True,
        )
        final_test_loader = DataLoader(
            final_test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
        )

        input_channels = len(final_train_df.columns) - len(EXCLUDE_COLS) + 1
        input_channels *= len(tickers)

        for name in MODELS_TO_RUN:
            model_name = f"benchmark_{dataset_name}_{name}"
            print(f"\n{model_name}")

            model = make_model(name, input_channels, len(tickers)).to(DEVICE)
            train_start, train_end = train_model(
                model,
                final_train_loader,
                EPOCHS,
            )

            infer_start = datetime.now()
            preds, actuals, masks, dates, tickers = predict(
                model,
                final_test_loader,
                tickers,
            )
            infer_end = datetime.now()

            valid = masks.astype(bool)
            rmse = float(np.sqrt(np.mean((preds[valid] - actuals[valid]) ** 2)))
            directional_accuracy = float(
                (np.sign(preds[valid]) == np.sign(actuals[valid])).mean()
            )
            rank_ic = calculate_rank_ic(preds, actuals, masks, dates)
            ls_spread = calculate_ls_spread(preds, actuals, masks)
            sharpe = calculate_sharpe(preds, actuals, masks)
            num_params = sum(p.numel() for p in model.parameters())

            print(
                f"FINAL | RMSE {rmse:.6f} | DirAcc {directional_accuracy:.2%} | "
                f"RankIC {rank_ic:.4f} | L/S {ls_spread:.6f} | Sharpe {sharpe:.4f}"
            )

            save_predictions(
                model_name,
                preds,
                actuals,
                masks,
                dates,
                tickers,
            )

            header = not RESULTS_FILE.exists()
            with RESULTS_FILE.open("a") as f:
                if header:
                    f.write(
                        "Model,RMSE,DirectionalAccuracy,RankIC,LS_Spread,"
                        "Sharpe,TrainTime,InferTime,NumParams\n"
                    )
                f.write(
                    f"{model_name},{rmse:.6f},{directional_accuracy:.6f},"
                    f"{rank_ic:.6f},{ls_spread:.6f},{sharpe:.6f},"
                    f"{train_end - train_start},{infer_end - infer_start},"
                    f"{num_params}\n"
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run_benchmarks()