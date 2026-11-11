import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import pipeline as pl
from rpeak_utils import count_matched, find_missed_rpeaks, remove_duplicate_rpeaks

st.set_page_config(page_title="ECG Dataset Builder", layout="wide")

PAGE_BG = "#FFFFFF"     # main page background
SIDEBAR_BG = "#D5DEEA"  # sidebar background
st.markdown(f"""
<style>
.stApp {{ background-color: {PAGE_BG}; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {{
    background-color: {SIDEBAR_BG};
}}
[data-testid="stSidebar"] {{ border-right: 1px solid #aebbd0; }}
</style>
""", unsafe_allow_html=True)


MODEL_DIR = Path(__file__).parent / "model"
# plot that reports its visible x-range (zoom) back to Python
zoom_plot = components.declare_component("zoom_plot", path=str(Path(__file__).parent / "components" / "zoom_plot"))


@st.cache_resource(show_spinner=False)
def cached_model(path, mtime, device, bf, ks, kn):
    import inference as inf          # needs PyTorch
    return inf.load_model(path, device, bf, ks, kn)


@st.cache_data
def cached_ncols(data):
    return pl.txt_num_columns(data)


@st.cache_data(show_spinner="Loading the ECG file...")
def cached_signal(data, col):
    return pl.read_signal_txt(data, col)


@st.cache_data
def cached_fs(data):
    return pl.infer_fs(data)


@st.cache_data
def cached_timelog(data):
    return pl.read_timelog(data)


@st.cache_data
def cached_ibis(data):
    return pl.read_ibis(data)


# ---------------- sidebar ----------------
st.sidebar.header("Settings")
FS = 250             # sampling rate of the ECG file (R-peak time (s) x FS = sample index)
window_sec = 4
stride_sec = 4
sigma_ms = 20
min_windows = 3      # a record with this many windows or fewer is skipped

st.sidebar.subheader("Detection settings")
threshold = st.sidebar.slider("Probability threshold", 0.05, 0.95, 0.5, 0.05)
min_dist = 72        # non-maximum-suppression window (samples)
tolerance = st.sidebar.number_input("Match tolerance (samples)", 0, 100, 5,
                                    help="A prediction is correct if it is within this many samples of a true peak.")
batch_size = 32

# model settings, only used when the model file is a state_dict
bf = 16
ks = 7
kn = 4

st.sidebar.subheader("R-peak correction settings")
gap_factor = st.sidebar.slider("Gap threshold (x local RR)", 1.3, 2.5, 1.6, 0.05,
                               help="An RR interval longer than this x its local median is searched for missed beats.")
snr = st.sidebar.slider("Sensitivity to small beats (SNR)", 2.0, 15.0, 6.0, 0.5,
                        help="Lower = finds weaker beats (but may add false ones).")
dup_factor = st.sidebar.slider("Duplicate threshold (x local RR)", 0.2, 0.8, 0.5, 0.05,
                               help="Peaks closer than this fraction of the local RR (and always closer than 0.2 s) "
                                    "are treated as double detections; the highest one is kept.")

# ---------------- upload ----------------
st.title("ECG Dataset Builder")
c1, c2, c3 = st.columns(3)
ecg_f = c1.file_uploader("ECG signal (250 Hz .txt)", type=["txt", "csv", "dat"], key="ecg")
log_f = c2.file_uploader("Timelog (.csv / .txt)", type=["csv", "txt"], key="timelog")
ibi_f = c3.file_uploader("IBI / R-peaks file (.txt / .csv)", type=["txt", "csv", "dat"], key="ibi")
if not (ecg_f and log_f and ibi_f):
    st.info("Upload the ECG file, the timelog and the IBI file to continue.")
    st.stop()

ecg_bytes = ecg_f.getvalue()
try:
    n_cols = cached_ncols(ecg_bytes)
    timelog = cached_timelog(log_f.getvalue())
    ibis = cached_ibis(ibi_f.getvalue())
except Exception as e:
    st.error(f"Could not read the files: {e}")
    st.stop()

with c1:
    col = 3# st.selectbox("Signal column", range(n_cols), index=3 if n_cols > 3 else n_cols - 1,
             #          format_func=lambda i: f"Column {i + 1}",
              #         help="Expected layout: column 1 = clock time, 2 and 3 = elapsed seconds, 4 = ECG signal.")
try:
    ecg = cached_signal(ecg_bytes, col)
except Exception as e:
    st.error(f"Column {col + 1} could not be read as numbers: {e}")
    st.stop()
inferred = cached_fs(ecg_bytes) if n_cols > 1 else float("nan")
if np.isfinite(inferred) and abs(inferred - FS) > 0.01 * FS:
    st.warning(f"The time column of the ECG file suggests about {inferred:.1f} Hz, "
               f"but the sampling rate in the sidebar is {FS} Hz.")
st.caption(f"Recording: {len(ecg) / FS / 60:.1f} min, {len(ecg):,} samples at {FS} Hz  |  "
           f"{len(timelog)} timelog records, {len(ibis):,} R-peaks in the IBI file")

# ---------------- plot (filled after the radio list so it can show the selection) ----------------
plot_box = st.container()

st.markdown("**Select a record** (label = Condition | Start | End, in minutes)")
labels = [f"{r.Condition}  |  Start: {r.Start:g}  |  End: {r.End:g}" for r in timelog.itertuples()]
sel = st.radio("Records", range(len(timelog)), index=None, format_func=lambda i: labels[i],
               key=f"rec_{log_f.name}_{log_f.size}", label_visibility="collapsed")

with plot_box:
    idx = pl.decimate_idx(ecg)
    fig = go.Figure(go.Scattergl(x=idx / FS, y=ecg[idx], mode="lines", name="ECG", line=dict(width=1)))
    if sel is not None:
        s_sec, e_sec = float(timelog.Start[sel]) * 60, float(timelog.End[sel]) * 60
        a, b = max(0, int(round(s_sec * FS))), min(len(ecg), int(round(e_sec * FS)) + 1)
        if a < b:
            seg = ecg[a:b]
            j = pl.decimate_idx(seg, 100_000)
            fig.add_vrect(x0=s_sec, x1=e_sec, fillcolor="orange", opacity=0.2, line_width=0, layer="below")
            fig.add_trace(go.Scattergl(x=(a + j) / FS, y=seg[j], mode="lines", name="Selected record",
                                       line=dict(width=1, color="darkorange")))
        else:
            st.warning("The selected record lies outside the recording.")
    fig.update_layout(title="ECG", xaxis_title="Time (s)", yaxis_title="Amplitude", hovermode="x",
                      template="plotly_white", height=450, uirevision=f"{sel}")
    st.plotly_chart(fig)

# ---------------- detect (builds the dataset for the selected record, then detects) ----------------
cur_key = (ecg_f.name, ecg_f.size, col, log_f.name, log_f.size, ibi_f.name, ibi_f.size,
           sel, FS, window_sec, stride_sec, sigma_ms)
st.divider()
st.subheader("Detect R-peaks")
model_files = sorted(p for p in MODEL_DIR.glob("*") if p.suffix.lower() in (".pt", ".pth", ".pkl"))
model_path = None
if model_files:
    model_path = st.selectbox("Trained model", model_files, format_func=lambda p: p.name)
else:
    st.info(f"No model found yet. Put your trained model (.pt / .pth) in this folder: {MODEL_DIR}")
det_key = cur_key + (model_path.name, threshold, min_dist, tolerance) if model_path else None
if st.button("🔍 Detect", type="primary", disabled=(sel is None or model_path is None)):
    try:
        with st.spinner("Building the dataset, loading the model and detecting R-peaks..."):
            rec = timelog.iloc[sel]
            X, Y, real, n_ann = pl.extract_record(ecg, ibis, rec.Start, rec.End, FS, window_sec, sigma_ms, stride_sec)
            if len(X) <= min_windows:
                st.session_state.pop("dataset", None)
                st.warning(f"This record gives only {len(X)} window(s); it is skipped (needs more than {min_windows}).")
            else:
                # kept for the next steps: st.session_state["dataset"]["X" | "Y" | "real_target"]
                st.session_state["dataset"] = dict(X=X, Y=Y, real_target=real, fs=FS, record=rec.to_dict(),
                                                   n_annotated=n_ann, key=cur_key)
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = cached_model(str(model_path), model_path.stat().st_mtime, device, bf, ks, kn)
                import inference as inf
                out = inf.run_detection(model, X, Y, real, device, threshold, min_dist, tolerance, batch_size, FS)
                st.session_state["detection"] = dict(pred=out["pred"], metrics=out["metrics"],
                                                     model=model_path.name, device=device, key=det_key)
    except ModuleNotFoundError as e:
        st.error(f"A required package is missing ({e.name}). For PyTorch run: pip install torch")
    except Exception as e:
        st.error(f"Detection failed: {type(e).__name__}: {e}")
if sel is None:
    st.caption("Select a record above to enable the Detect button.")

ds = st.session_state.get("dataset")
det = st.session_state.get("detection")
if det and ds and det["key"] == det_key:
    mt = det["metrics"]
    r1 = st.columns(5)
    r1[0].metric("True positives", mt["TP"])
    r1[1].metric("False positives", mt["FP"])
    r1[2].metric("False negatives", mt["FN"])
    r1[3].metric("Precision", f"{mt['precision']:.4f}")
    r1[4].metric("Recall", f"{mt['recall']:.4f}")
    r2 = st.columns(5)
    r2[0].metric("F1", f"{mt['f1']:.4f}")
    r2[1].metric("DER (%)", f"{mt['der']:.2f}")
    r2[2].metric("MAE (ms)", f"{mt['mae_ms']:.2f}")
    r2[3].metric("95th percentile error (ms)", f"{mt['p95_ms']:.2f}")
    r2[4].metric("Windows", len(det["pred"]))
    st.caption(f"Model: {det['model']}  |  device: {det['device']}")

    # rebuild the continuous signal of the selected record from its windows
    X = ds["X"]
    n_win, L = X.shape
    stride = int(stride_sec * FS)
    sig = np.full((n_win - 1) * stride + L, np.nan, dtype=np.float32)
    for w in range(n_win):
        sig[w * stride: w * stride + L] = X[w]
    to_global = lambda lists: np.concatenate(
        [np.asarray(r, dtype=np.int64) + w * stride for w, r in enumerate(lists)] or [np.empty(0, np.int64)])
    true_g, pred_g = to_global(ds["real_target"]), to_global(det["pred"])
    t0 = int(round(float(ds["record"]["Start"]) * 60 * FS)) / FS
    tt = lambda i: t0 + np.asarray(i) / FS

    # ---------------- inspect & correct R-peaks ----------------
    st.divider()
    st.subheader("Inspect & correct R-peaks")
    st.caption("Works on the predicted R-peaks above: removes duplicate (too close) peaks, adds missed ones, "
               "and compares the old and the new IBIs.")
    EMPTY = np.array([], dtype=np.int64)
    ecg_rec = np.nan_to_num(sig, nan=float(np.nanmedian(sig)))
    pred0, true0 = np.unique(pred_g), np.unique(true_g)

    corr = st.session_state.get("correction")
    if not corr or corr["key"] != det_key:          # new detection -> start from scratch
        corr = dict(key=det_key, added=EMPTY, removed=EMPTY, manual=EMPTY, msg="")
        st.session_state["correction"] = corr
    corr.setdefault("manual", EMPTY)

    cb1, cb2, _ = st.columns([2, 1, 5])
    if cb1.button("🛠️ Auto-correct R-peaks", type="primary",
                  help="Removes duplicate (too close) peaks, then adds missed ones."):
        manual_now = corr["manual"]
        dups = remove_duplicate_rpeaks(ecg_rec, pred0, FS, dup_factor)      # never removes a manual peak
        kept = np.setdiff1d(pred0, dups)
        # the manual peaks count as existing peaks, so the gaps they fill are not searched again
        new, gaps = find_missed_rpeaks(ecg_rec, np.union1d(kept, manual_now), FS, gap_factor, snr)
        new = np.setdiff1d(new, np.union1d(pred0, manual_now))
        corr.update(added=new, removed=dups,
                    msg=(f"Removed {len(dups)} duplicate peak(s); added {len(new)} missed peak(s) in {gaps} gap(s)"
                         + (f"; kept your {len(manual_now)} manual peak(s)." if len(manual_now) else ".")
                         if len(dups) or len(new) else "Nothing to correct with the current settings."))
    if cb2.button("Reset", key="reset_correction"):
        corr.update(added=EMPTY, removed=EMPTY, manual=EMPTY, msg="")
    if corr["msg"]:
        st.info(corr["msg"])

    added, removed, manual = corr["added"], corr["removed"], corr["manual"]
    final = np.union1d(np.union1d(np.setdiff1d(pred0, removed), added), manual)

    cm = st.columns(6)
    cm[0].metric("Predicted peaks", len(pred0))
    cm[1].metric("Removed duplicates", len(removed))
    cm[2].metric("Added peaks", len(added))
    cm[3].metric("Added manually", len(manual))
    cm[4].metric("Total after correction", len(final))
    if len(true0):
        before, after = count_matched(true0, pred0, tolerance), count_matched(true0, final, tolerance)
        cm[5].metric(f"True peaks matched (±{tolerance} samples)", f"{after}/{len(true0)}", delta=after - before)

    kk = pl.decimate_idx(ecg_rec)
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.07)
    fig3.add_trace(go.Scattergl(x=tt(kk), y=ecg_rec[kk], mode="lines", name="ECG", line=dict(width=1)), row=1, col=1)
    fig3.add_trace(go.Scattergl(x=tt(pred0), y=ecg_rec[pred0], mode="markers", name="Predicted R-peaks",
                                marker=dict(size=5, color="red")), row=1, col=1)
    if len(true0):
        fig3.add_trace(go.Scattergl(x=tt(true0), y=ecg_rec[true0], mode="markers", name="True R-peaks",
                                    marker=dict(size=5, color="green")), row=1, col=1)
    if len(removed):
        fig3.add_trace(go.Scattergl(x=tt(removed), y=ecg_rec[removed], mode="markers", name="Removed duplicates",
                                    marker=dict(size=9, color="black", symbol="x")), row=1, col=1)
    if len(added):
        fig3.add_trace(go.Scattergl(x=tt(added), y=ecg_rec[added], mode="markers", name="Added R-peaks",
                                    marker=dict(size=9, color="royalblue", symbol="diamond",
                                                line=dict(width=1, color="black"))), row=1, col=1)
    if len(manual):
        fig3.add_trace(go.Scattergl(x=tt(manual), y=ecg_rec[manual], mode="markers", name="Added manually",
                                    marker=dict(size=11, color="darkorange", symbol="star",
                                                line=dict(width=1, color="black"))), row=1, col=1)
    if len(pred0) > 1:   # IBI = time between consecutive R-peaks, drawn at the time of the later beat
        fig3.add_trace(go.Scattergl(x=tt(pred0)[1:], y=np.diff(tt(pred0)), mode="lines+markers",
                                    name="IBI (original predicted)", line=dict(width=1.5, color="red"),
                                    marker=dict(size=3)), row=2, col=1)
    if len(added) or len(removed) or len(manual):
        fig3.add_trace(go.Scattergl(x=tt(final)[1:], y=np.diff(tt(final)), mode="lines+markers",
                                    name="IBI (corrected)", line=dict(width=1.5, color="royalblue"),
                                    marker=dict(size=3)), row=2, col=1)
    fig3.update_layout(title="Predicted R-peaks, corrections and IBI", hovermode="x", template="plotly_white", height=780,
                       uirevision=str(det_key))   # keep the zoom when the page reruns
    fig3.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig3.update_yaxes(title_text="IBI (s)", row=2, col=1)
    fig3.update_xaxes(title_text="Time (s)", row=2, col=1)
    zoom_val = zoom_plot(fig_json=fig3.to_json(), height=780, key="corr_chart", default=None)

    # ---- add an R-peak by hand ----
    n_samp = len(sig)
    st.markdown("**Add an R-peak manually**")
    st.caption("Zoom into the part of the plot you want (drag on the plot): From and To are filled with the first and "
               "last sample of the visible part. Then enter the sample where the R-peak should be and press Add. "
               f"Samples are counted inside the selected record: sample = (time in s - {t0:g}) x {FS}.")
    st.session_state.setdefault("manual_from", 0)
    st.session_state.setdefault("manual_to", n_samp - 1)
    if zoom_val and "x0" in zoom_val:
        s0 = int(np.clip(np.ceil(round((zoom_val["x0"] - t0) * FS, 4)), 0, n_samp - 1))
        s1 = int(np.clip(np.floor(round((zoom_val["x1"] - t0) * FS, 4)), 0, n_samp - 1))
        if (s0, s1) != st.session_state.get("last_zoom"):      # only when the zoom changed
            st.session_state.update(last_zoom=(s0, s1), manual_from=s0, manual_to=max(s0, s1))
    for k in ("manual_from", "manual_to", "manual_loc"):
        st.session_state[k] = int(min(max(st.session_state.get(k, 0), 0), n_samp - 1))

    mc = st.columns([1, 1, 1, 1])
    m_from = mc[0].number_input("From (sample)", 0, n_samp - 1, step=1, key="manual_from")
    m_to = mc[1].number_input("To (sample)", 0, n_samp - 1, step=1, key="manual_to")
    m_loc = mc[2].number_input("R-peak location (sample)", 0, n_samp - 1, step=1, key="manual_loc")
    mc[3].markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if mc[3].button("Add", type="primary", key="manual_add"):
        if m_from >= m_to:
            st.session_state["manual_msg"] = ("error", "'From' must be smaller than 'To'.")
        elif not (m_from <= m_loc <= m_to):
            st.session_state["manual_msg"] = ("error", f"The R-peak location must lie between {m_from} and {m_to}.")
        elif m_loc in final:
            st.session_state["manual_msg"] = ("warning", f"There is already an R-peak at sample {m_loc}.")
        else:
            near = final[np.abs(final - m_loc) < 0.2 * FS]          # closer than 200 ms to another peak
            corr["manual"] = np.union1d(manual, [m_loc]).astype(np.int64)
            note = (f" It is only {int(np.abs(near - m_loc).min())} samples from another R-peak - "
                    "check that it is not a duplicate." if len(near) else "")
            st.session_state["manual_msg"] = ("success", f"Added an R-peak at sample {m_loc} "
                                                         f"({tt(m_loc):.3f} s); the IBIs were updated.{note}")
            st.rerun()
        st.rerun()
    if "manual_msg" in st.session_state:
        kind, text = st.session_state["manual_msg"]
        getattr(st, kind)(text)

    # downloads: always present and closed; nothing downloads unless you click
    ran = bool(corr["msg"]) or len(manual) > 0
    with st.expander("⬇️ Download corrected R-peaks and IBIs"):
        if not ran:
            st.caption("Run 'Auto-correct R-peaks' first to enable the downloads.")
        rec_start = int(round(float(ds["record"]["Start"]) * 60 * FS))
        pk_df = pd.DataFrame({"sample_index_in_record": final, "sample_index_in_recording": final + rec_start,
                              "time_s": tt(final)})
        ibi_df = pd.DataFrame({"sample_index_in_record": final[1:], "time_s": tt(final)[1:],
                               "ibi_s": np.diff(tt(final))})
        dc = st.columns(4)
        dc[0].download_button("R-peaks (CSV)", pk_df.to_csv(index=False), file_name="corrected_rpeaks.csv",
                              mime="text/csv", disabled=not ran)
        dc[1].download_button("IBIs (CSV)", ibi_df.to_csv(index=False), file_name="corrected_ibis.csv",
                              mime="text/csv", disabled=not ran)
        st.caption("The R-peak CSV gives sample indices inside the selected record and inside the whole recording; the IBI CSV gives sample index, time and IBI in seconds.")
        if len(added):
            st.caption(f"{len(added)} added peaks:")
            st.dataframe(pd.DataFrame({"sample_index_in_record": added, "time_s": tt(added)}))
        if len(manual):
            st.caption(f"{len(manual)} peaks added manually:")
            st.dataframe(pd.DataFrame({"sample_index_in_record": manual, "time_s": tt(manual)}))
        if len(removed):
            st.caption(f"{len(removed)} removed duplicate peaks:")
            st.dataframe(pd.DataFrame({"sample_index_in_record": removed, "time_s": tt(removed)}))
elif det:
    st.caption("The model, selection or settings changed since the last detection: press Detect again.")
