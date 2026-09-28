import os
from pathlib import Path

import torch

# dirs
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "processed_data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "trained_models"
MLFLOW_DIR = ROOT_DIR / "mlruns"

# dataset
TRAIN_SIZE = 0.75
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.10
GAP_DAYS = 90
WAVELET_WINDOW = 30
TDA_WINDOW = 30
STRIDE = 5
FINAL_WARMUP = max(GAP_DAYS, WAVELET_WINDOW, TDA_WINDOW) + 1

# training
WINDOW_SIZE = 90
EPOCHS = 50
BATCH_SIZE = 128
BENCHMARK_LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5

# default hyperparameters
D_MODEL = 128
N_LAYERS = 4
D_STATE = 64
EXPAND = 4
HEAD_DIM = 64
DROPOUT = 0.0

N_TRIALS = 20
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["default", "wavelet", "tda", "tda_wavelet"]
ARCHITECTURES = ["base", "glu", "mask", "both"]

# mlflow
MLFLOW_EXPERIMENT = "ssm-stock-prediction"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DIR.as_uri())

# plots
TICKER = "AAPL"