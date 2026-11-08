"""Model loading, inference and evaluation (taken from your pipeline)."""
import io
import pickle
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import ecg_unet

# ---------------- loading a model saved with torch.save(model) ----------------
_OUR_CLASSES = {"Attention1d", "ODConv1d", "ODConvBlock", "ConvBlock", "UpBlock", "ECGUNet"}


class _RemapUnpickler(pickle.Unpickler):
    """A model saved from a notebook refers to `__main__.ECGUNet`; point those names at ecg_unet.py."""

    def find_class(self, module, name):
        if name.split(".")[0] in _OUR_CLASSES:
            obj = ecg_unet
            for part in name.split("."):
                obj = getattr(obj, part)
            return obj
        return super().find_class(module, name)


_PICKLE = types.ModuleType("remap_pickle")
_PICKLE.__dict__.update({k: v for k, v in vars(pickle).items() if not k.startswith("__")})
_PICKLE.Unpickler = _RemapUnpickler
_PICKLE.load = lambda f, **kw: _RemapUnpickler(f, **kw).load()
_PICKLE.loads = lambda b, **kw: _RemapUnpickler(io.BytesIO(b), **kw).load()


def load_model(path, device="cpu", base_filters=8, kernel_size=7, kernel_num=1):
    """Load a full pickled model, or a state_dict (then ECGUNet is rebuilt from the given settings)."""
    obj = torch.load(path, map_location=device, weights_only=False, pickle_module=_PICKLE)
    if isinstance(obj, dict):
        state = obj.get("model_state_dict") or obj.get("state_dict") or obj
        model = ecg_unet.ECGUNet(1, 1, base_filters, kernel_size, kernel_num)
        model.load_state_dict(state)
    else:
        model = obj
    return model.to(device).eval()


# ---------------- your dataset / inference / metrics code ----------------
class ECGRpeakDataset(Dataset):
    def __init__(self, input, target, real_target=None, normalize=True):
        self.input, self.target, self.real_target, self.normalize = input, target, real_target, normalize

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        signal = np.asarray(self.input[idx], dtype=np.float32)
        if self.normalize:
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)   # z-score every window
        x = torch.from_numpy(signal)
        y = torch.from_numpy(np.asarray(self.target[idx]))
        real_y = torch.from_numpy(np.asarray(self.real_target[idx]))
        return x, y, real_y


def ecg_collate_fn(batch):
    signals = torch.stack([s[0] for s in batch])
    heatmaps = torch.stack([s[1] for s in batch])
    real_rpeaks = [s[2] for s in batch]          # variable length -> keep as a list
    return signals, heatmaps, real_rpeaks


def predict_rpeaks(model, x, threshold=0.5, min_dist=72, device="cpu"):
    """x: [B, 1, L] -> list of sorted R-peak index arrays (threshold + greedy non-maximum suppression)."""
    model.eval()
    with torch.no_grad():
        heatmap = torch.sigmoid(model(x.to(device)))
    heatmap = heatmap.squeeze(1).cpu().numpy()
    results = []
    for prob in heatmap:
        candidates = np.where(prob > threshold)[0]
        if len(candidates) == 0:
            results.append(np.array([], dtype=np.int64))
            continue
        peaks, suppressed = [], np.zeros(len(prob), dtype=bool)
        for idx in candidates[np.argsort(prob[candidates])[::-1]]:
            if suppressed[idx]:
                continue
            peaks.append(idx)
            suppressed[max(0, idx - min_dist): min(len(prob), idx + min_dist + 1)] = True
        results.append(np.sort(np.array(peaks, dtype=np.int64)))
    return results


def compute_metrics(pred_peaks, true_peaks, tolerance=0, fs=250):
    TP = FP = FN = 0
    temporal_loc_errs = []                       # milliseconds
    for pred, true in zip(pred_peaks, true_peaks):
        pred, true = np.asarray(pred), np.asarray(true)
        matched_true = np.zeros(len(true), dtype=bool)
        for p in pred:
            if len(true) == 0:
                FP += 1
                continue
            distances = np.abs(true - p)
            idx = np.argmin(distances)
            if distances[idx] <= tolerance and not matched_true[idx]:
                TP += 1
                matched_true[idx] = True
                temporal_loc_errs.append(int(distances[idx]) / fs * 1000)
            else:
                FP += 1
        FN += int(np.sum(~matched_true))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1,
            "temporal_err": temporal_loc_errs}


def run_detection(model, X, Y, real_target, device="cpu", threshold=0.5, min_dist=72, tolerance=5,
                  batch_size=16, fs=250):
    """Detect R-peaks in every window and evaluate against the real R-peaks."""
    loader = DataLoader(ECGRpeakDataset(X, Y, real_target), batch_size=batch_size, shuffle=False,
                        collate_fn=ecg_collate_fn)
    preds, reals = [], []
    for signals, _, real in loader:
        preds.extend(predict_rpeaks(model, signals.unsqueeze(1), threshold, min_dist, device))
        reals.extend(real)

    TP = FP = FN = 0
    mae_w, p95_w = [], []                        # per-window values, averaged like in your evaluation code
    for p, r in zip(preds, reals):
        res = compute_metrics([p], [r], tolerance, fs)
        TP, FP, FN = TP + res["TP"], FP + res["FP"], FN + res["FN"]
        errs = res["temporal_err"]
        mae_w.append(float(np.mean(errs)) if errs else 0.0)
        p95_w.append(float(np.percentile(errs, 95)) if errs else 0.0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    der = (FP + FN) / (TP + FN) * 100 if (TP + FN) > 0 else (100.0 if FP > 0 else 0.0)
    metrics = dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1, der=der,
                   mae_ms=float(np.mean(mae_w)) if mae_w else 0.0,
                   p95_ms=float(np.mean(p95_w)) if p95_w else 0.0)
    return {"pred": preds, "metrics": metrics}
