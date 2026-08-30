# train.py — full-epochs training, WeightedMSE (+optional L1 & mean-match),
#            optional shape-only loss + small time-shift tolerance,
#            per-file train/val split (with gap, only when val_ratio>0),
#            device-aware DataLoader,
#            TRAIN (+ optional VAL) DC-align + [0,1] plots, CSV/NPZ export,
#            + per-epoch per-channel metrics CSV,
#            + save ALL targets & predictions (train[/val]) as NPZ.

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt

from model import TCNRegressor
from dataset import SequenceDataset, find_excel_files, INPUT_COLS, OUTPUT_COLS


# -------- Weighted loss -------
class WeightedMSE(nn.Module):
    """MSE with per-channel weights = 1/sigma^2 (sigma from training targets)."""
    def __init__(self, y_sd: np.ndarray):
        super().__init__()
        w = 1.0 / (np.maximum(y_sd, 1e-8) ** 2)
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32))  # (C,)
    def forward(self, yhat, y):
        # yhat,y: (B,C,L)
        return ((yhat - y) ** 2 * self.w.view(1, -1, 1)).mean()


def train_one_epoch(model, loader, device, loss_fn, optimizer, grad_clip=0.5):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(1, n)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (Y, Yhat) in STANDARDIZED space with shape (N,C,L)."""
    model.eval()
    ys, yhats = [], []
    for x, y in loader:
        x = x.to(device)
        yhat = model(x)
        ys.append(y.cpu().numpy()); yhats.append(yhat.cpu().numpy())
    if not ys: return None, None
    return np.concatenate(ys, 0), np.concatenate(yhats, 0)  # (N,C,L)


# ---- utilities for shape-only & shift-tolerance ----
def zscore_along_time(t: torch.Tensor, eps: float = 1e-8):
    # t: (B,C,L)
    mu = t.mean(dim=-1, keepdim=True)
    sd = t.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return (t - mu) / sd


def shift_tensor(x: torch.Tensor, s: int):
    # x: (B,C,L); s>0 shift right by s (replicate pad), s<0 shift left.
    if s == 0:
        return x
    L = x.size(-1)
    if s > 0:
        y = F.pad(x, (s, 0), mode='replicate')[..., -L:]
    else:
        y = F.pad(x, (0, -s), mode='replicate')[..., :L]
    return y


@torch.no_grad()
def compute_metrics_phys(model, loader, device, y_mu_1d: np.ndarray, y_sd_1d: np.ndarray):
    """
    Compute channel-wise MAE, RMSE, and R2 in physical units over the entire loader.
    Returns: mae(C,), rmse(C,), r2(C,)
    """
    model.eval()
    C = int(len(y_sd_1d))
    mae_sum = np.zeros(C, dtype=np.float64)
    mse_sum = np.zeros(C, dtype=np.float64)
    sum_y = np.zeros(C, dtype=np.float64)
    sum_y2 = np.zeros(C, dtype=np.float64)
    count = 0

    y_sd_t = torch.tensor(y_sd_1d, dtype=torch.float32, device=device).view(1, -1, 1)
    y_mu_t = torch.tensor(y_mu_1d, dtype=torch.float32, device=device).view(1, -1, 1)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)

        diff_phys = (yhat - y) * y_sd_t        # (B,C,L)
        mae_sum += diff_phys.abs().sum(dim=(0, 2)).cpu().numpy()
        mse_sum += (diff_phys ** 2).sum(dim=(0, 2)).cpu().numpy()

        y_phys = y * y_sd_t + y_mu_t
        sum_y  += y_phys.sum(dim=(0, 2)).cpu().numpy()
        sum_y2 += (y_phys ** 2).sum(dim=(0, 2)).cpu().numpy()

        count += y.shape[0] * y.shape[2]

    denom = max(1, count)
    mae = mae_sum / denom
    rmse = np.sqrt(mse_sum / denom)
    mean_y = sum_y / denom
    ss_tot = sum_y2 - 2 * mean_y * sum_y + (mean_y ** 2) * denom
    ss_res = mse_sum
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return mae, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='train_data')
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dropout', type=float, default=0.05)
    ap.add_argument('--channels', type=str, default='64,64,128,128')
    ap.add_argument('--kernel_size', type=int, default=5)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, default='checkpoints')
    ap.add_argument('--plot_windows', type=int, default=3)
    ap.add_argument('--plot_out', type=str, default='viz_train')
    ap.add_argument('--l1_ratio', type=float, default=0.2, help='total = (1-a)*WMSE + a*L1')
    ap.add_argument('--mean_match', type=float, default=0.1, help='weight of mean-match term to learn DC bias')
    # Options
    ap.add_argument('--shape_only_loss', action='store_true',
                    help='Apply a per-window z-score to compare shape only (removing DC offset and scale); commonly used with shift_tolerance.')
    ap.add_argument('--shift_tolerance', type=int, default=0,
                    help='Allow a temporal-shift search of +/-S frames in the loss; 0 disables it. Suggested initial range: 2-4.')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ===== Files =====
    files = find_excel_files(Path(args.data_dir))
    assert len(files) > 0, "No Excel files were found in the data directory."
    print("[Files]", *[f.name for f in files], sep="\n  - ")

    # ===== Fit scalers ON ALL FILES (simple & stable) =====
    scaler_ds = SequenceDataset(files, args.seq_len, args.stride, fit_scaler=True)
    x_mu, x_sd, y_mu, y_sd = scaler_ds.x_mu, scaler_ds.x_sd, scaler_ds.y_mu, scaler_ds.y_sd

    # ===== Per-file split into train/val windows =====
    use_val = (args.val_ratio is not None) and (args.val_ratio > 0)
    gap = int(np.ceil(args.seq_len / max(1, args.stride))) if use_val else 0  # Leave a gap only when a validation set is used
    print(f"[Split] per-file val_ratio={args.val_ratio:.2f}, gap={gap} windows, use_val={use_val}")

    train_subsets, val_subsets, report = [], [], []
    for f in files:
        ds_f = SequenceDataset([f], args.seq_len, args.stride,
                               x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, fit_scaler=False)
        n = len(ds_f)
        if n == 0:
            report.append((f.name, 0, 0, 0)); continue

        if use_val:
            n_val = int(round(n * args.val_ratio))
            n_val = min(max(n_val, 1), n)  # At least 1 and at most n
            train_end = max(0, n - n_val - gap)
            train_idx = np.arange(0, train_end, dtype=int)
            val_idx   = np.arange(n - n_val, n, dtype=int)
            if train_idx.size > 0: train_subsets.append(Subset(ds_f, train_idx))
            if val_idx.size   > 0: val_subsets.append(Subset(ds_f, val_idx))
            report.append((f.name, n, train_idx.size, val_idx.size))
        else:
            # Use all windows for training without leaving a gap
            train_idx = np.arange(0, n, dtype=int)
            train_subsets.append(Subset(ds_f, train_idx))
            report.append((f.name, n, train_idx.size, 0))

    assert len(train_subsets) > 0, "The training set is empty. Please check the data or parameters."
    if use_val:
        assert len(val_subsets) > 0, "The validation set is empty. Please reduce val_ratio or gap."

    print("[Per-file windows]")
    for name, n, nt, nv in report:
        print(f"  - {name}: total={n}, train={nt}, val={nv}")

    # ===== Device & DataLoader (pin_memory only on CUDA) =====
    use_cuda = torch.cuda.is_available()
    use_mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device   = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
    pin_mem  = (device.type == "cuda")
    print(f"[Device] {device} | pin_memory={pin_mem}")

    train_ds = ConcatDataset(train_subsets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)

    val_loader = None
    if use_val:
        val_ds = ConcatDataset(val_subsets)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin_mem)

    # ===== Model =====
    channels = [int(x) for x in args.channels.split(',') if x.strip()]
    model = TCNRegressor(in_channels=len(INPUT_COLS), out_channels=len(OUTPUT_COLS),
                         channels=channels, kernel_size=args.kernel_size,
                         dropout=args.dropout).to(device)

    # ===== Loss / Optim / Sched =====
    wmse = WeightedMSE(y_sd).to(device)

    def loss_fn(yhat, y):
        # yhat,y: (B,C,L) in standardized space
        if args.shape_only_loss:
            y_z = zscore_along_time(y)
            if args.shift_tolerance > 0:
                best = None
                for s in range(-args.shift_tolerance, args.shift_tolerance + 1):
                    cand = zscore_along_time(shift_tensor(yhat, s))
                    v = ((cand - y_z) ** 2).mean().detach().item()
                    if (best is None) or (v < best[0]): best = (v, s)
                yhat_z = zscore_along_time(shift_tensor(yhat, best[1]))
            else:
                yhat_z = zscore_along_time(yhat)
            loss = ((yhat_z - y_z) ** 2).mean()
            if args.l1_ratio > 0:
                loss = (1.0 - args.l1_ratio) * loss + args.l1_ratio * (yhat_z - y_z).abs().mean()
            return loss
        else:
            if args.shift_tolerance > 0:
                best = None
                for s in range(-args.shift_tolerance, args.shift_tolerance + 1):
                    cand = shift_tensor(yhat, s)
                    v = wmse(cand, y).detach().item()
                    if (best is None) or (v < best[0]): best = (v, s)
                yhat = shift_tensor(yhat, best[1])
            wm = wmse(yhat, y)
            if args.l1_ratio > 0:
                wm = (1.0 - args.l1_ratio) * wm + args.l1_ratio * (yhat - y).abs().mean()
            if args.mean_match > 0:
                mu_hat = yhat.mean(dim=(0,2))
                mu_gt  = y.mean(dim=(0,2))
                mm = ((mu_hat - mu_gt) ** 2).mean()
                wm = wm + args.mean_match * mm
            return wm

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=5, min_lr=1e-6)

    # Save the scalers for inverse standardization
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'scalers.npz', x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd,
             inputs=np.array(INPUT_COLS), outputs=np.array(OUTPUT_COLS))

    # ===== Train =====
    best_metric = float('inf')
    epoch_rows = []  # (epoch, split, channel, mae, rmse, r2)

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        if use_val:
            va = evaluate(model, val_loader, device, loss_fn)
            scheduler.step(va)
        else:
            va = None
            scheduler.step(tr)

        # Channel-wise training/validation metrics in physical units
        mae_tr, rmse_tr, r2_tr = compute_metrics_phys(model, train_loader, device, y_mu, y_sd)
        if use_val:
            mae_va, rmse_va, r2_va = compute_metrics_phys(model, val_loader, device, y_mu, y_sd)

        for ci, cname in enumerate(OUTPUT_COLS):
            epoch_rows.append([epoch, 'train', cname, float(mae_tr[ci]), float(rmse_tr[ci]), float(r2_tr[ci])])
            if use_val:
                epoch_rows.append([epoch, 'val',   cname, float(mae_va[ci]), float(rmse_va[ci]), float(r2_va[ci])])

        if use_val:
            print(f'Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f} | '
                  f'lr {optimizer.param_groups[0]["lr"]:.2e}')
            cur_metric = va
        else:
            print(f'Epoch {epoch:03d} | train {tr:.6f} | '
                  f'lr {optimizer.param_groups[0]["lr"]:.2e}')
            cur_metric = tr  # Use training loss as the selection criterion when validation is disabled

        if cur_metric < best_metric - 1e-9:
            best_metric = cur_metric
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_loss': tr, 'val_loss': va if use_val else None,
                        'config': vars(args)},
                       out_dir / 'best_model.pt')

    # Save channel-wise metrics for each epoch
    epoch_csv = out_dir / 'epoch_channel_metrics.csv'
    with open(epoch_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'split', 'channel', 'mae_phys', 'rmse_phys', 'r2'])
        writer.writerows(epoch_rows)
    print('Saved per-epoch channel metrics to:', epoch_csv)

    torch.save(model.state_dict(), out_dir / 'last_model_state.pt')
    print('Done. Best metric (normalized domain):', best_metric,
          '(val used)' if use_val else '(train used)')

    # ===== Plot helpers =====
    def plot_and_dump(prefix: str, Y_s: np.ndarray, YH_s: np.ndarray, y_mu_: np.ndarray, y_sd_: np.ndarray):
        if Y_s is None:
            print(f'No {prefix} windows to plot.');
            return
        # Convert back to physical units and transpose to (N,L,C)
        Y  = np.transpose(Y_s,  (0,2,1)) * y_sd_ + y_mu_
        YH = np.transpose(YH_s, (0,2,1)) * y_sd_ + y_mu_

        viz_dir = Path(args.plot_out if prefix=='train' else args.plot_out + '_val')
        viz_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = viz_dir / 'csv_per_window'; csv_dir.mkdir(parents=True, exist_ok=True)

        num_windows = int(min(args.plot_windows, Y.shape[0]))
        L = Y.shape[1]; C = Y.shape[2]
        x_axis = np.arange(L)

        t_phys_all    = np.zeros((num_windows, L, C), dtype=np.float32)
        p_phys_dc_all = np.zeros((num_windows, L, C), dtype=np.float32)
        t_norm_all    = np.zeros((num_windows, L, C), dtype=np.float32)
        p_norm_all    = np.zeros((num_windows, L, C), dtype=np.float32)
        dc_shift_all  = np.zeros((num_windows, C), dtype=np.float32)

        for w in range(num_windows):
            for ci, cname in enumerate(OUTPUT_COLS):
                t = Y[w, :, ci].astype(float)
                p = YH[w, :, ci].astype(float)

                # DC alignment and joint normalization
                mu_t = float(t.mean()); mu_p = float(p.mean())
                dc_shift = (mu_t - mu_p)
                p_aligned = p + dc_shift

                vmin = float(min(t.min(), p_aligned.min()))
                vmax = float(max(t.max(), p_aligned.max()))
                denom = max(vmax - vmin, 1e-12)
                t_n = (t - vmin) / denom
                p_n = (p_aligned - vmin) / denom

                # CSV
                out_csv = csv_dir / f'{prefix}_window{w}_{cname}.csv'
                data_mat = np.column_stack([x_axis, t, p_aligned, t_n, p_n])
                header = 'sample,target_phys,pred_phys_dc_aligned,target_norm,pred_norm'
                np.savetxt(out_csv, data_mat, delimiter=',', header=header, comments='', fmt='%.8f')

                # PNG
                plt.figure()
                plt.plot(x_axis, t_n, label='target (norm)')
                plt.plot(x_axis, p_n, label='pred (DC-aligned, norm)')
                plt.xlabel('sample'); plt.ylabel('normalized (0–1)')
                ttl = f'{cname} | DC shift = {dc_shift:.3f} (phys units)'
                if prefix != 'train': ttl = f'(VAL) {ttl}'
                plt.title(ttl)
                plt.legend(); plt.tight_layout()
                plt.savefig(viz_dir / f'{prefix}_window{w}_{cname}_dc_aligned_norm.png', dpi=150)
                plt.close()

                # Aggregate data for NPZ export
                t_phys_all[w, :, ci]    = t
                p_phys_dc_all[w, :, ci] = p_aligned
                t_norm_all[w, :, ci]    = t_n
                p_norm_all[w, :, ci]    = p_n
                dc_shift_all[w, ci]     = dc_shift

        np.savez(
            viz_dir / f'{prefix}_windows_data.npz',
            x=np.arange(L, dtype=np.int32),
            channel_names=np.array(OUTPUT_COLS, dtype=object),
            t_phys=t_phys_all,
            p_phys_dc=p_phys_dc_all,
            t_norm=t_norm_all,
            p_norm=p_norm_all,
            dc_shift=dc_shift_all,
            meta=np.array({
                'note': f'First num_windows windows saved (DC-aligned and normalized) for {prefix}.',
                'num_windows': num_windows
            }, dtype=object)
        )
        print(f'Saved {prefix} plots to:', viz_dir)
        print(f'Saved {prefix} CSVs to:', csv_dir)
        print(f'Saved {prefix} NPZ to:', viz_dir / f'{prefix}_windows_data.npz')

    # ===== Load best model and plot TRAIN (+ optional VAL) =====
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state'])

    scal = np.load(out_dir / 'scalers.npz', allow_pickle=True)
    y_mu_np = scal['y_mu'].reshape(1,1,-1); y_sd_np = scal['y_sd'].reshape(1,1,-1)

    # Generate predictions for the full dataset in standardized space
    Y_tr_s, YH_tr_s = collect_predictions(model, train_loader, device)
    plot_and_dump('train', Y_tr_s, YH_tr_s, y_mu_np, y_sd_np)

    if use_val and val_loader is not None:
        Y_val_s, YH_val_s = collect_predictions(model, val_loader, device)
        plot_and_dump('val',   Y_val_s, YH_val_s, y_mu_np, y_sd_np)

    # ===== Save all targets and predictions in physical units =====
    def save_all(prefix: str, Y_s: np.ndarray, YH_s: np.ndarray):
        if Y_s is None: return
        Y  = np.transpose(Y_s,  (0,2,1)) * y_sd_np + y_mu_np  # (N,L,C)
        YH = np.transpose(YH_s, (0,2,1)) * y_sd_np + y_mu_np
        np.savez(
            out_dir / f'all_{prefix}_targets_preds.npz',
            y_true=Y.astype(np.float32),
            y_pred=YH.astype(np.float32),
            channel_names=np.array(OUTPUT_COLS, dtype=object),
            x=np.arange(Y.shape[1], dtype=np.int32)
        )
        print(f"Saved ALL {prefix} targets & preds to:", out_dir / f'all_{prefix}_targets_preds.npz')

    save_all('train', Y_tr_s, YH_tr_s)
    if use_val and val_loader is not None:
        save_all('val',   Y_val_s, YH_val_s)


if __name__ == '__main__':
    main()
