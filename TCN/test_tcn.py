# test_tcn.py


import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import TCNRegressor
from dataset import INPUT_COLS, OUTPUT_COLS


# ================== Path settings ==================
TEST_DATA_DIR = Path(r"F:/chengxu/DM-Tac/test_data")
CKPT_DIR      = Path(r"F:/chengxu/DM-Tac/checkpoints")
OUT_DIR       = Path(r"F:/chengxu/DM-Tac/test_outputs")

MODEL_PATH  = CKPT_DIR / "best_model.pt"
SCALER_PATH = CKPT_DIR / "scalers.npz"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================== Parameters ==================
SEQ_LEN = 128
STRIDE  = 32
BATCH_SIZE = 16


DEFAULT_CHANNELS = [64, 64, 128, 128]
DEFAULT_KERNEL_SIZE = 5
DEFAULT_DROPOUT = 0.05


def find_excel_files(data_dir: Path):
    files = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(glob.glob(str(data_dir / ext)))
    files = sorted([Path(f) for f in files])
    if len(files) == 0:
        raise FileNotFoundError(f"No Excel or CSV files were found in {data_dir}.")
    return files


def read_table(file_path: Path):
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def load_scalers(scaler_path: Path):
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    scal = np.load(scaler_path, allow_pickle=True)

    x_mu = scal["x_mu"].astype(np.float32)
    x_sd = scal["x_sd"].astype(np.float32)
    y_mu = scal["y_mu"].astype(np.float32)
    y_sd = scal["y_sd"].astype(np.float32)

    return x_mu, x_sd, y_mu, y_sd


def build_model_from_checkpoint(model_path: Path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    config = ckpt.get("config", {})

    channels_str = config.get("channels", None)
    if channels_str is not None:
        channels = [int(x) for x in channels_str.split(",") if x.strip()]
    else:
        channels = DEFAULT_CHANNELS

    kernel_size = int(config.get("kernel_size", DEFAULT_KERNEL_SIZE))
    dropout = float(config.get("dropout", DEFAULT_DROPOUT))

    model = TCNRegressor(
        in_channels=len(INPUT_COLS),
        out_channels=len(OUTPUT_COLS),
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print("[Model]")
    print("  channels   =", channels)
    print("  kernel_size=", kernel_size)
    print("  dropout    =", dropout)

    return model


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: [N, C]
    return: mae, rmse, r2, each [C]
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff = y_pred - y_true

    mae = np.mean(np.abs(diff), axis=0)
    rmse = np.sqrt(np.mean(diff ** 2, axis=0))

    ss_res = np.sum(diff ** 2, axis=0)
    mean_y = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - mean_y) ** 2, axis=0)

    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    return mae, rmse, r2


@torch.no_grad()
def predict_one_file(file_path: Path, model, device, x_mu, x_sd, y_mu, y_sd):
    """
    Perform sliding-window prediction on a single Excel/CSV file and average
    the predictions from overlapping windows back to the original frame-level sequence.

    Returns:
        df_out: Frame-level targets and predictions aligned with the original rows.
        metrics_rows: Evaluation metrics, with one row for each channel.
    """

    df = read_table(file_path)

    # ================== Validate Columns ==================
    missing_inputs = [c for c in INPUT_COLS if c not in df.columns]
    missing_outputs = [c for c in OUTPUT_COLS if c not in df.columns]

    if missing_inputs:
        raise ValueError(f"{file_path.name} is missing the following input columns: {missing_inputs}")

    if missing_outputs:
        raise ValueError(f"{file_path.name} is missing the last three ground-truth columns: {missing_outputs}")

    # Retain only rows with complete input and output data
    use_cols = INPUT_COLS + OUTPUT_COLS
    df_use = df[use_cols].copy()
    valid_mask = np.isfinite(df_use.to_numpy(dtype=np.float64)).all(axis=1)
    df_use = df_use.loc[valid_mask].reset_index(drop=True)

    if len(df_use) < SEQ_LEN:
        raise ValueError(
            f"{file_path.name} contains {len(df_use)} valid rows, fewer than SEQ_LEN={SEQ_LEN}"
        )

    X_raw = df_use[INPUT_COLS].to_numpy(dtype=np.float32)
    Y_true = df_use[OUTPUT_COLS].to_numpy(dtype=np.float32)

    # ================== Normalize Inputs ==================
    X_std = (X_raw - x_mu) / x_sd

    N = len(df_use)
    C_out = len(OUTPUT_COLS)

    pred_sum = np.zeros((N, C_out), dtype=np.float64)
    pred_cnt = np.zeros((N, C_out), dtype=np.float64)

    starts = list(range(0, N - SEQ_LEN + 1, STRIDE))

    # Ensure that the final window covers the end of the sequence
    last_start = N - SEQ_LEN
    if starts[-1] != last_start:
        starts.append(last_start)

    # ================== Batch Prediction ==================
    for b0 in range(0, len(starts), BATCH_SIZE):
        batch_starts = starts[b0:b0 + BATCH_SIZE]

        xb = []
        for s in batch_starts:
            xw = X_std[s:s + SEQ_LEN, :]      # [L, Cin]
            xb.append(xw.T)                   # [Cin, L]

        xb = np.stack(xb, axis=0).astype(np.float32)  # [B, Cin, L]
        xb_t = torch.from_numpy(xb).to(device)

        yh_std = model(xb_t).cpu().numpy()            # [B, Cout, L]
        yh_std = np.transpose(yh_std, (0, 2, 1))      # [B, L, Cout]

        # Inverse normalization
        yh_phys = yh_std * y_sd.reshape(1, 1, -1) + y_mu.reshape(1, 1, -1)

        # Average overlapping-window predictions to obtain frame-level predictions
        for bi, s in enumerate(batch_starts):
            e = s + SEQ_LEN
            pred_sum[s:e, :] += yh_phys[bi]
            pred_cnt[s:e, :] += 1.0

    Y_pred = pred_sum / np.maximum(pred_cnt, 1e-12)

    # ================== metrics ==================
    mae, rmse, r2 = compute_metrics(Y_true, Y_pred)

    metrics_rows = []
    for ci, cname in enumerate(OUTPUT_COLS):
        metrics_rows.append({
            "file": file_path.name,
            "channel": cname,
            "mae_phys": mae[ci],
            "rmse_phys": rmse[ci],
            "r2": r2[ci],
            "n_samples": N,
            "seq_len": SEQ_LEN,
            "stride": STRIDE
        })

    # ================== Export Frame-Level Predictions ==================
    df_out = pd.DataFrame()
    df_out["sample"] = np.arange(N)

    if "time" in df.columns:
        # valid_mask identifies valid rows in the original DataFrame
        df_out["time"] = df.loc[valid_mask, "time"].to_numpy()
    else:
        df_out["time"] = np.arange(N)

    for ci, cname in enumerate(OUTPUT_COLS):
        df_out[f"{cname}_true"] = Y_true[:, ci]
        df_out[f"{cname}_pred"] = Y_pred[:, ci]
        df_out[f"{cname}_error"] = Y_pred[:, ci] - Y_true[:, ci]

    return df_out, metrics_rows, Y_true, Y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    files = find_excel_files(TEST_DATA_DIR)
    print("[Test files]")
    for f in files:
        print("  -", f.name)

    x_mu, x_sd, y_mu, y_sd = load_scalers(SCALER_PATH)
    model = build_model_from_checkpoint(MODEL_PATH, device)

    all_metrics = []
    all_true = []
    all_pred = []
    all_file_ids = []

    for fi, file_path in enumerate(files):
        print("\n[Test]", file_path.name)

        df_pred, metrics_rows, y_true, y_pred = predict_one_file(
            file_path, model, device, x_mu, x_sd, y_mu, y_sd
        )

        base = file_path.stem

        out_pred_csv = OUT_DIR / f"{base}_test_predictions.csv"
        df_pred.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")

        print("  saved:", out_pred_csv)

        for row in metrics_rows:
            all_metrics.append(row)

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_file_ids.append(np.full((len(y_true),), fi, dtype=np.int32))

    # ================== Save Metrics for Each File ==================
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = OUT_DIR / "test_metrics_per_file.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    # ================== Overall Metrics ==================
    Y_true_all = np.concatenate(all_true, axis=0)
    Y_pred_all = np.concatenate(all_pred, axis=0)
    file_id_all = np.concatenate(all_file_ids, axis=0)

    mae, rmse, r2 = compute_metrics(Y_true_all, Y_pred_all)

    all_rows = []
    for ci, cname in enumerate(OUTPUT_COLS):
        all_rows.append({
            "file": "ALL_TEST_FILES",
            "channel": cname,
            "mae_phys": mae[ci],
            "rmse_phys": rmse[ci],
            "r2": r2[ci],
            "n_samples": len(Y_true_all),
            "seq_len": SEQ_LEN,
            "stride": STRIDE
        })

    metrics_all_df = pd.DataFrame(all_rows)
    metrics_all_csv = OUT_DIR / "test_metrics_all.csv"
    metrics_all_df.to_csv(metrics_all_csv, index=False, encoding="utf-8-sig")

    # ================== Save as NPZ for Subsequent Plotting ==================
    out_npz = OUT_DIR / "all_test_targets_preds.npz"
    np.savez(
        out_npz,
        y_true=Y_true_all.astype(np.float32),
        y_pred=Y_pred_all.astype(np.float32),
        file_id=file_id_all,
        channel_names=np.array(OUTPUT_COLS, dtype=object),
        input_cols=np.array(INPUT_COLS, dtype=object),
    )

    print("\nCompleted")
    print("Prediction results for each file:", OUT_DIR)
    print("Per-file metrics:", metrics_csv)
    print("Overall metrics:", metrics_all_csv)
    print("Overall target/prediction NPZ:", out_npz)


if __name__ == "__main__":
    main()