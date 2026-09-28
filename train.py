from datetime import datetime

import mlflow
import numpy as np
import optuna
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from config import (
    ARCHITECTURES,
    BATCH_SIZE,
    DATASETS,
    DEVICE,
    EPOCHS,
    GAP_DAYS,
    GRAD_CLIP,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    N_TRIALS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from model import CrossSectionalDataset, collate_batch, model_factory


EXCLUDE_COLS = {"Date", "Ticker", "target_next_ret"}
FINAL_RESULTS_FILE = RESULTS_DIR / "final_results.csv"
FINAL_RESULTS_HEADERS = (
    "DateTime,Model,RMSE,DirectionalAccuracy,RankIC,LS_Spread,Sharpe,"
    "TrainTime,InferTime,TotalTime,NumParams\n"
)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TradingLoss(nn.Module):
    """RMSE, correlation and directional loss."""

    def __init__(self, alpha=1.0, beta=0.5, gamma=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, mask):
        valid = mask > 0
        preds = preds[valid]
        targets = targets[valid]

        if preds.numel() == 0:
            return preds.sum() * 0

        rmse = torch.sqrt(torch.mean((preds - targets) ** 2) + 1e-8)
        p_dev = preds - preds.mean()
        t_dev = targets - targets.mean()
        corr = torch.sum(p_dev * t_dev) / (
            torch.sqrt(torch.sum(p_dev ** 2) + 1e-8)
            * torch.sqrt(torch.sum(t_dev ** 2) + 1e-8)
        )
        corr_loss = 1.0 - corr
        direction_loss = torch.mean(torch.relu(-preds * targets))

        return (
            self.alpha * rmse
            + self.beta * corr_loss
            + self.gamma * direction_loss
        )


def split_train_validation_test(df):
    """Split data chronologically with gaps between splits."""
    dates = np.array(
        sorted(df["Date"].unique().to_list()),
        dtype="datetime64[D]",
    )
    n = len(dates)

    train_end_idx = max(1, int(n * TRAIN_SIZE)) - 1
    train_end = dates[train_end_idx]

    validation_start = train_end + np.timedelta64(GAP_DAYS, "D")
    validation_end_idx = min(
        n - 1,
        int(n * (TRAIN_SIZE + VALIDATION_SIZE)) - 1,
    )
    validation_end = dates[validation_end_idx]
    test_start = validation_end + np.timedelta64(GAP_DAYS, "D")

    train = df.filter(pl.col("Date") <= train_end)
    validation = df.filter(
        (pl.col("Date") >= validation_start)
        & (pl.col("Date") <= validation_end)
    )
    test = df.filter(pl.col("Date") >= test_start)

    if train.is_empty() or validation.is_empty() or test.is_empty():
        raise ValueError("Train/validation/test split produced an empty split.")

    print(
        f"  Split dates: train <= {train_end}, "
        f"validation {validation_start} to {validation_end}, "
        f"test >= {test_start}"
    )
    print(
        f"  Rows: train={train.height:,}, "
        f"validation={validation.height:,}, test={test.height:,}"
    )
    return train, validation, test


def clean_data(df):
    """Remove invalid inputs and targets."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    conditions = [
        pl.col("target_next_ret").is_not_null()
        & pl.col("target_next_ret").is_finite()
    ]
    conditions.extend(
        pl.col(c).is_not_null() & pl.col(c).is_finite()
        for c in feature_cols
    )

    condition = conditions[0]
    for extra in conditions[1:]:
        condition &= extra

    return df.filter(condition)


def scale_features(train_df, other_dfs):
    """Fit the scaler on train data only."""
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
    scaler = RobustScaler()
    train_values = scaler.fit_transform(
        train_df.select(feature_cols).to_numpy()
    )

    def rebuild(original, values):
        return pl.concat(
            [
                original.select(["Date", "Ticker"]),
                pl.DataFrame(values, schema=feature_cols),
                original.select(["target_next_ret"]),
            ],
            how="horizontal",
        )

    return rebuild(train_df, train_values), [
        rebuild(
            df,
            scaler.transform(df.select(feature_cols).to_numpy()),
        )
        for df in other_dfs
    ]


def make_loaders(train_df, validation_df, test_df, tickers):
    train_ds = CrossSectionalDataset(
        train_df,
        train_df,
        WINDOW_SIZE,
        tickers,
    )
    val_ds = CrossSectionalDataset(
        train_df,
        validation_df,
        WINDOW_SIZE,
        tickers,
    )
    test_history = pl.concat([train_df, validation_df], how="vertical")
    test_ds = CrossSectionalDataset(
        test_history,
        test_df,
        WINDOW_SIZE,
        tickers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    return train_loader, val_loader, test_loader


def _masked_metrics(preds, actuals, masks):
    valid = masks.astype(bool)
    preds = preds[valid]
    actuals = actuals[valid]

    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    directional_accuracy = float(
        (np.sign(preds) == np.sign(actuals)).mean()
    )
    return rmse, directional_accuracy


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0

    for x, y, mask, *_ in loader:
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()
        mask = mask.to(DEVICE).float()

        with torch.set_grad_enabled(training):
            preds = model(x)
            loss = criterion(preds, y, mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / max(batches, 1)


def predict(model, loader, tickers):
    model.eval()
    predictions, actuals, masks, dates = [], [], [], []

    with torch.no_grad():
        for x, y, mask, batch_dates, _ in loader:
            preds = model(x.to(DEVICE).float()).cpu().numpy()
            predictions.append(preds)
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


def validation_objective(
    trial,
    architecture,
    train_loader,
    val_loader,
    input_dim,
):
    params = {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-5,
            5e-2,
            log=True,
        ),
        "epochs": trial.suggest_int("epochs", 10, EPOCHS),
    }

    with mlflow.start_run(
        run_name=f"trial_{trial.number}",
        nested=True,
    ):
        mlflow.log_params(params)

        model = model_factory(
            architecture,
            input_dim,
            d_model=params["d_model"],
            n_layers=params["n_layers"],
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params["lr"],
            total_steps=max(len(train_loader) * params["epochs"], 1),
        )
        criterion = TradingLoss()

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(params["epochs"]):
            model.train()

            for x, y, mask, *_ in train_loader:
                x = x.to(DEVICE).float()
                y = y.to(DEVICE).float()
                mask = mask.to(DEVICE).float()

                optimizer.zero_grad()
                loss = criterion(model(x), y, mask)

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            val_loss = run_epoch(model, val_loader, criterion)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise optuna.TrialPruned()

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= 4:
                    break

        mlflow.log_metric("best_val_loss", best_val)
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_val


def tune_architecture(architecture, train_loader, val_loader, input_dim):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: validation_objective(
            trial,
            architecture,
            train_loader,
            val_loader,
            input_dim,
        ),
        n_trials=N_TRIALS,
    )
    return study.best_params


def train_final_model(model, train_loader, epochs, model_name, test_loader=None):
    criterion = TradingLoss()
    # Per-epoch log read by plot_training.py. The test loss is for monitoring
    # only and is never used for model selection.
    log_path = RESULTS_DIR / f"{model_name}.csv"
    log_path.write_text("Epoch,TrainLoss,TestLoss\n")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model._experiment_lr,
        weight_decay=model._experiment_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=model._experiment_lr,
        total_steps=max(len(train_loader) * epochs, 1),
    )

    start = datetime.now()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for x, y, mask, *_ in train_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()
            mask = mask.to(DEVICE).float()

            optimizer.zero_grad()
            loss = criterion(model(x), y, mask)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(
            f"  [{model_name}] epoch {epoch + 1:02d}/{epochs} | "
            f"loss {avg_loss:.5f}"
        )
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        test_loss = (
            run_epoch(model, test_loader, criterion)
            if test_loader is not None
            else float("nan")
        )
        with log_path.open("a") as f:
            f.write(f"{epoch + 1},{avg_loss:.8f},{test_loss:.8f}\n")

    return start, datetime.now()


def calculate_rank_ic(preds, actuals, masks, dates):
    values = []

    for i, date in enumerate(dates):
        valid = masks[i].astype(bool)
        if valid.sum() < 2:
            continue

        x = preds[i][valid]
        y = actuals[i][valid]
        x_rank = np.argsort(np.argsort(x)).astype(float)
        y_rank = np.argsort(np.argsort(y)).astype(float)
        values.append(np.corrcoef(x_rank, y_rank)[0, 1])

    return float(np.nanmean(values)) if values else 0.0


def calculate_ls_spread(preds, actuals, masks):
    spreads = []

    for pred, actual, mask in zip(preds, actuals, masks):
        valid = mask.astype(bool)
        if valid.sum() < 2:
            continue

        pred = pred[valid]
        actual = actual[valid]
        k = max(1, int(len(pred) * 0.10))
        order = np.argsort(pred)
        spreads.append(
            actual[order[-k:]].mean()
            - actual[order[:k]].mean()
        )

    return float(np.mean(spreads)) if spreads else 0.0


def calculate_sharpe(preds, actuals, masks):
    strategy_returns = []

    for pred, actual, mask in zip(preds, actuals, masks):
        valid = mask.astype(bool)
        strategy_returns.extend(
            (np.sign(pred[valid]) * actual[valid]).tolist()
        )

    strategy_returns = np.asarray(strategy_returns)

    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return 0.0

    return float(
        strategy_returns.mean() / strategy_returns.std()
    )


def save_predictions(model_name, preds, actuals, masks, dates, tickers):
    rows = []

    for i, date in enumerate(dates):
        for j, ticker in enumerate(tickers):
            if masks[i, j] == 0:
                continue

            rows.append(
                (date, ticker, preds[i, j], actuals[i, j])
            )

    pl.DataFrame(
        rows,
        schema=[
            "Date",
            "Ticker",
            "Predicted_Return",
            "Actual_Return",
        ],
        orient="row",
    ).write_csv(
        RESULTS_DIR / f"{model_name}_final_predictions.csv"
    )


def run_experiment():
    set_seed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for dataset_name in DATASETS:
        print(
            f"\n{'=' * 60}\n"
            f"DATASET: {dataset_name}\n"
            f"{'=' * 60}"
        )

        full_data = pl.read_parquet(
            PROCESSED_DIR / f"full_{dataset_name}.parquet"
        )
        full_data = clean_data(full_data)

        train_raw, val_raw, test_raw = split_train_validation_test(
            full_data
        )

        tickers = sorted(train_raw["Ticker"].unique().to_list())
        train_df, [val_df, test_df] = scale_features(
            train_raw,
            [val_raw, test_raw],
        )

        train_loader, val_loader, test_loader = make_loaders(
            train_df,
            val_df,
            test_df,
            tickers,
        )
        input_dim = len(train_df.columns) - len(EXCLUDE_COLS) + 1

        for architecture in ARCHITECTURES:
            if (dataset_name, architecture) == ("tda_wavelet", "base"):
                print(f"  [SKIP] {dataset_name}/{architecture}")
                continue

            model_name = f"mamba_{dataset_name}_{architecture}"
            print(f"\n{model_name}")

            with mlflow.start_run(run_name=model_name):
                best_params = tune_architecture(
                    architecture,
                    train_loader,
                    val_loader,
                    input_dim,
                )
                print(f"  Best parameters: {best_params}")

                final_train_raw = pl.concat(
                    [train_raw, val_raw],
                    how="vertical",
                )
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

                model = model_factory(
                    architecture,
                    input_dim,
                    d_model=best_params["d_model"],
                    n_layers=best_params["n_layers"],
                ).to(DEVICE)

                model._experiment_lr = best_params["lr"]
                model._experiment_weight_decay = best_params["weight_decay"]

                mlflow.log_params(best_params)
                mlflow.log_params({
                    "window_size": WINDOW_SIZE,
                    "gap_days": GAP_DAYS,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                })

                train_start, train_end = train_final_model(
                    model,
                    final_train_loader,
                    best_params["epochs"],
                    model_name,
                    final_test_loader,
                )

                infer_start = datetime.now()
                preds, actuals, masks, dates, tickers = predict(
                    model,
                    final_test_loader,
                    tickers,
                )
                infer_end = datetime.now()

                rmse, directional_accuracy = _masked_metrics(
                    preds,
                    actuals,
                    masks,
                )
                rank_ic = calculate_rank_ic(
                    preds,
                    actuals,
                    masks,
                    dates,
                )
                ls_spread = calculate_ls_spread(
                    preds,
                    actuals,
                    masks,
                )
                sharpe = calculate_sharpe(
                    preds,
                    actuals,
                    masks,
                )
                num_params = sum(
                    p.numel() for p in model.parameters()
                )

                print(
                    f"FINAL | RMSE {rmse:.6f} | "
                    f"DirAcc {directional_accuracy:.2%} | "
                    f"RankIC {rank_ic:.4f} | "
                    f"L/S {ls_spread:.6f} | "
                    f"Sharpe {sharpe:.4f}"
                )

                mlflow.log_metrics({
                    "rmse": rmse,
                    "directional_accuracy": directional_accuracy,
                    "rank_ic": rank_ic,
                    "ls_spread": ls_spread,
                    "sharpe": sharpe,
                    "num_params": num_params,
                })

                checkpoint = MODELS_DIR / f"{model_name}_final.pth"
                torch.save(model.state_dict(), checkpoint)
                mlflow.log_artifact(str(checkpoint))

                save_predictions(
                    model_name,
                    preds,
                    actuals,
                    masks,
                    dates,
                    tickers,
                )
                mlflow.log_artifact(
                    str(
                        RESULTS_DIR
                        / f"{model_name}_final_predictions.csv"
                    )
                )

                total_time = infer_end - train_start
                train_time = train_end - train_start
                infer_time = infer_end - infer_start

                if not FINAL_RESULTS_FILE.exists():
                    FINAL_RESULTS_FILE.write_text(
                        FINAL_RESULTS_HEADERS
                    )

                with FINAL_RESULTS_FILE.open("a") as f:
                    f.write(
                        f"{datetime.now()},{model_name},{rmse:.6f},"
                        f"{directional_accuracy:.6f},{rank_ic:.6f},"
                        f"{ls_spread:.6f},{sharpe:.6f},"
                        f"{train_time},{infer_time},{total_time},"
                        f"{num_params}\n"
                    )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run_experiment()