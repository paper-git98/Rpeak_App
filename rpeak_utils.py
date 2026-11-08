"""Helpers for finding R-peaks that the model missed (including very small beats)."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def _one_pass(s, r, floor, pred, fs, gap_factor, snr, refr):
    rr = np.diff(pred)
    med = np.array([np.median(rr[max(0, i - 6): i + 7]) for i in range(len(rr))])  # local RR
    half = max(2, int(0.06 * fs))
    new, gaps = [], 0
    for i in np.where(rr > gap_factor * med)[0]:
        a, b = pred[i], pred[i + 1]
        inner = r[a + refr: b - refr]
        if len(inner) < 3:
            continue
        n = max(1, int(round(rr[i] / med[i])) - 1)      # beats that should fit in this gap
        step = rr[i] / (n + 1)
        noise = max(np.median(inner), floor)             # LOCAL noise level -> works for tiny beats
        found = 0
        for k in range(1, n + 1):
            c = a + k * step
            lo, hi = int(max(c - 0.4 * step, a + refr)), int(min(c + 0.4 * step, b - refr))
            if hi - lo < 3:
                continue
            j = lo + int(np.argmax(r[lo:hi]))            # strongest QRS-like activity in this slot
            if r[j] < snr * noise:
                continue                                 # nothing beat-like here (pure noise / dropout)
            k0, k1 = max(0, j - half), min(len(s), j + half)
            p = k0 + int(np.argmax(s[k0:k1]))            # snap to the R top
            if a + refr <= p <= b - refr:
                new.append(p)
                found += 1
        gaps += found > 0
    return new, gaps


def find_missed_rpeaks(ecg, pred, fs, gap_factor=1.6, snr=6.0, refractory_s=0.2):
    """Find long RR gaps between predicted peaks and add peaks inside them.

    The beat detector inside a gap is judged against the noise level OF THAT GAP,
    so low-amplitude beats next to huge ones are still found.
    Returns (new_indices, number_of_gaps_filled).
    """
    ecg = np.asarray(ecg, float)
    pred = np.unique(np.asarray(pred, int))
    if len(pred) < 4:
        return np.array([], dtype=int), 0
    base = np.median(ecg)
    sign = 1.0 if np.median(ecg[pred]) >= base else -1.0   # handles inverted leads
    s = sign * (ecg - base)
    w1, w2 = max(1, int(0.02 * fs)), max(4, int(0.2 * fs))
    r = np.abs(uniform_filter1d(s, w1) - uniform_filter1d(s, w2))  # QRS-scale deflections only
    floor = 0.0005 * np.median(r[pred])                   # ignores perfectly flat lines
    refr = int(refractory_s * fs)

    cur, all_new, total = pred, [], 0
    for _ in range(2):                                    # 2nd pass fills slots missed in the 1st
        new, g = _one_pass(s, r, floor, cur, fs, gap_factor, snr, refr)
        new = np.setdiff1d(np.unique(new), cur)
        if len(new) == 0:
            break
        all_new.extend(new.tolist())
        total += g
        cur = np.union1d(cur, new)
    return np.array(sorted(all_new), dtype=int), total


def count_matched(true, pred, tol):
    """How many true peaks have a predicted peak within `tol` samples."""
    true, pred = np.asarray(true), np.sort(np.asarray(pred))
    if len(pred) == 0 or len(true) == 0:
        return 0
    j = np.searchsorted(pred, true)
    left = np.abs(true - pred[np.clip(j - 1, 0, len(pred) - 1)])
    right = np.abs(true - pred[np.clip(j, 0, len(pred) - 1)])
    return int(np.sum(np.minimum(left, right) <= tol))


def remove_duplicate_rpeaks(ecg, pred, fs, dup_factor=0.5, refractory_s=0.2):
    """Find predicted peaks that are double detections of one beat.

    Peaks closer than max(refractory_s, dup_factor x local RR) to their neighbour form a cluster;
    the peak nearest the R apex (highest, polarity-aware) is kept, the others are returned for removal.
    """
    ecg = np.asarray(ecg, float)
    pred = np.unique(np.asarray(pred, int))
    if len(pred) < 3:
        return np.array([], dtype=int)
    base = np.median(ecg)
    sign = 1.0 if np.median(ecg[pred]) >= base else -1.0
    s = sign * (ecg - base)
    rr = np.diff(pred)
    med = np.array([np.median(rr[max(0, i - 6): i + 7]) for i in range(len(rr))])
    close = rr < np.maximum(refractory_s * fs, dup_factor * med)
    removed, i = [], 0
    while i < len(rr):
        if not close[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rr) and close[j + 1]:
            j += 1
        grp = pred[i: j + 2]                       # peaks i..j+1 are one cluster
        keep = grp[np.argmax(s[grp])]
        removed.extend(grp[grp != keep].tolist())
        i = j + 1
    return np.array(removed, dtype=int)
