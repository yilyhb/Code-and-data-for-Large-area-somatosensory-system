# dataset.py — Excel loader + robust scaling (median/IQR), 9 input features
import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# 只保留这 9 个输入特征（你要求的）
INPUT_COLS = ['Tsum','TCoP_x','TCoP_y','TIxx','TIyy','TIxy','Tmax','value2','value3','value4','value5','value6','value7']
# INPUT_COLS = ['value2','value3','value4','value5','value6','value7']
# INPUT_COLS = ['value8','value9','value10']
# INPUT_COLS = ['Tsum','TCoP_x','TCoP_y','TIxx','TIyy','TIxy','Tmax']
OUTPUT_COLS = ['hip_flexion_r_moment','knee_angle_r_moment','ankle_angle_r_moment']
# INPUT_COLS = ['Tsum','TCoP_x','TCoP_y','TIxx','TIyy','TIxy','value8','value9','value10']
# OUTPUT_COLS = ['knee_angle_r_moment']

def _to_numpy(cols):
    try:
        return cols.to_numpy(dtype=np.float32)
    except Exception:
        return np.asarray(cols.values, dtype=np.float32)

def standardize_fit(X):
    # 用中位数+IQR，抗“多数为0、少数有峰”的列
    mu = np.nanmedian(X, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    sd = q75 - q25
    sd[sd < 1e-8] = 1.0
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

class SequenceDataset(Dataset):
    def __init__(self, files, seq_len, stride,
                 x_mu=None, x_sd=None, y_mu=None, y_sd=None, fit_scaler=False):
        super().__init__()
        self.files = files
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.index = []
        self._tables = []

        for f in self.files:
            df = pd.read_excel(f)
            miss_in = [c for c in INPUT_COLS if c not in df.columns]
            miss_out = [c for c in OUTPUT_COLS if c not in df.columns]
            if miss_in or miss_out:
                raise ValueError(f'{getattr(f,"name",str(f))} missing: inputs={miss_in}, outputs={miss_out}')
            df = df[INPUT_COLS + OUTPUT_COLS].dropna().reset_index(drop=True)
            self._tables.append(df)

        if fit_scaler:
            Xc = np.concatenate([_to_numpy(t[INPUT_COLS]) for t in self._tables], axis=0)
            Yc = np.concatenate([_to_numpy(t[OUTPUT_COLS]) for t in self._tables], axis=0)
            self.x_mu, self.x_sd = standardize_fit(Xc)
            self.y_mu, self.y_sd = standardize_fit(Yc)
        else:
            if any(v is None for v in (x_mu, x_sd, y_mu, y_sd)):
                raise ValueError('Must supply scalers when fit_scaler=False')
            self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd

        for fi, df in enumerate(self._tables):
            n = len(df)
            if n < self.seq_len:
                continue
            for s in range(0, n - self.seq_len + 1, self.stride):
                self.index.append((fi, s))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]
        df = self._tables[fi]
        x = _to_numpy(df.loc[s:s+self.seq_len-1, INPUT_COLS])
        y = _to_numpy(df.loc[s:s+self.seq_len-1, OUTPUT_COLS])
        x = standardize_apply(x, self.x_mu, self.x_sd).astype(np.float32)
        y = standardize_apply(y, self.y_mu, self.y_sd).astype(np.float32)
        return torch.from_numpy(x.T), torch.from_numpy(y.T)  # (C,L)

def find_excel_files(data_dir):
    files = []
    for ext in ('*.xlsx','*.xls'):
        pattern = os.path.join(str(data_dir), ext)
        for p in glob.glob(pattern):
            files.append(Path(p))
    files = sorted(files, key=lambda p: str(p))
    if not files:
        raise FileNotFoundError(f'No Excel files found in {data_dir}')
    return files
