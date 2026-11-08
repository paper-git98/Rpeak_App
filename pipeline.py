"""Loading helpers + the dataset-creation process (same logic as the original create_dataset)."""
import io

import numpy as np
import pandas as pd

TIMELOG_COLS = ["Visit Date", "Segment", "Condition", "Start", "End"]
IBI_COLS = ["chron_time", "time_start_1", "time_start_2", "ibi_ms"]


# ---------------- file loading ----------------
def txt_num_columns(data: bytes) -> int:
    """Number of whitespace-separated columns in the first line of the signal file."""
    return len(data[:65536].decode("latin-1").strip().splitlines()[0].split())


def read_signal_txt(data: bytes, col: int = 3) -> np.ndarray:
    """Read one column of the 250 Hz signal file (whitespace separated, no header)."""
    df = pd.read_csv(io.BytesIO(data), sep=r"\s+", header=None, usecols=[col], dtype=np.float32)
    return df.iloc[:, 0].to_numpy()


def infer_fs(data: bytes, time_col: int = 1, n: int = 2000) -> float:
    """Sampling rate implied by the elapsed-time column (looks at the first rows only)."""
    t = pd.read_csv(io.BytesIO(data), sep=r"\s+", header=None, usecols=[time_col], nrows=n,
                    dtype=np.float64).iloc[:, 0].to_numpy()
    step = np.median(np.diff(t)) if len(t) > 1 else 0
    return 1.0 / step if step > 0 else float("nan")


def read_timelog(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    if df.shape[1] != len(TIMELOG_COLS):
        raise ValueError(f"The timelog needs {len(TIMELOG_COLS)} columns {TIMELOG_COLS}, found {df.shape[1]}.")
    df.columns = TIMELOG_COLS
    for c in ("Start", "End"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Start", "End"]).reset_index(drop=True)


def read_ibis(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), sep=r"\s+", header=None)
    if df.shape[1] != len(IBI_COLS):
        raise ValueError(f"The IBI file needs {len(IBI_COLS)} columns {IBI_COLS}, found {df.shape[1]}.")
    df.columns = IBI_COLS
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def decimate_idx(y, max_points=200_000):
    """Indices of a min/max decimation (keeps spikes) so very long recordings stay plottable."""
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    k = int(np.ceil(n / (max_points // 2)))
    m = n // k * k
    blocks = np.asarray(y[:m]).reshape(-1, k)
    base = np.arange(0, m, k)
    return np.sort(np.stack([base + blocks.argmin(1), base + blocks.argmax(1)], 1), axis=1).ravel()


# ---------------- dataset creation (your process) ----------------
def make_rpeak_target(signal_length, rpeaks, fs=250, sigma_ms=20):
    """Gaussian target with a peak at every R-peak location."""
    target = np.zeros(signal_length, dtype=np.float32)
    sigma = max(1, int(round((sigma_ms / 1000) * fs)))
    radius = int(round(4 * sigma))
    for rp in np.asarray(rpeaks, dtype=int):
        if rp < 0 or rp >= signal_length:
            continue
        left, right = max(0, rp - radius), min(signal_length, rp + radius + 1)
        idx = np.arange(left, right)
        gaussian = np.exp(-0.5 * ((idx - rp) / sigma) ** 2).astype(np.float32)
        target[left:right] = np.maximum(target[left:right], gaussian)
    return target


def create_windows(ecg, rpeaks, fs=250, window_sec=4, sigma_ms=20, stride_sec=4):
    """Cut the ECG into windows; returns X (n, window), Y gaussian targets (n, window), R-peak list per window."""
    window_size, stride = int(window_sec * fs), int(stride_sec * fs)
    X, Y, real_targets = [], [], []
    for start in range(0, len(ecg) - window_size + 1, stride):
        end = start + window_size
        local_rpeaks = rpeaks[(rpeaks >= start) & (rpeaks < end)] - start
        X.append(np.asarray(ecg[start:end]).astype(np.float32))
        Y.append(make_rpeak_target(window_size, local_rpeaks, fs, sigma_ms))
        real_targets.append(np.asarray(local_rpeaks, dtype=np.int64).reshape(-1))
    X = np.array(X) if X else np.empty((0, window_size), np.float32)
    Y = np.array(Y) if Y else np.empty((0, window_size), np.float32)
    return X, Y, real_targets          # real_targets: list of int64 arrays, one per window row


def extract_record(ecg, ibis_df, start_min, end_min, fs=250, window_sec=4, sigma_ms=20, stride_sec=4):
    """One selected timelog record -> (X, Y, real_target, number_of_annotated_rpeaks)."""
    start_sec, end_sec = float(start_min) * 60, float(end_min) * 60
    t1 = ibis_df.iloc[:, 1]
    rpeaks = (ibis_df.loc[(t1 > start_sec) & (t1 < end_sec)]["time_start_1"] * fs).astype(int).to_numpy()
    startsample, endsample = int(np.rint(start_sec * fs)), int(np.rint(end_sec * fs))
    sig = ecg[startsample:endsample + 1]
    rp_abs = rpeaks[(rpeaks >= startsample) & (rpeaks < endsample)]
    rp_rel = rp_abs - startsample                      # relative to the start of the record
    X, Y, real = create_windows(sig, rp_rel, fs, window_sec, sigma_ms, stride_sec)
    return X, Y, real, len(rpeaks)
