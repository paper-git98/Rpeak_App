# ECG Dataset Builder
    pip install -r requirements.txt
    pip install torch            # only needed for the Detect step (see requirements-detect.txt)
    streamlit run app.py

1. Upload the 250 Hz ECG .txt, the timelog and the IBI file, then pick a record.
2. Put your trained model (.pt / .pth) in the `model` folder, choose it and press Detect.
   Detect builds the windowed dataset of the selected record (X, Y, real_target) and then detects R-peaks.
   Full models (torch.save(model, ...)) load as they are; state_dict files are rebuilt with the
   "Model settings" in the sidebar.
3. "Inspect & correct R-peaks" removes duplicate peaks, adds missed ones and compares the old and
   corrected IBIs; the corrected peaks / IBIs can be downloaded.
Results stay in st.session_state: "dataset" (X, Y, real_target), "detection" (pred, metrics),
"correction" (added, removed).

To confirm the model code is identical to yours:
    python verify_architecture.py path/to/your_original_file.py

Manual R-peak add: zoom into a part of the correction plot (drag on it) - From / To are filled with the first and
last sample of the visible part. Type the sample of the R-peak and press Add. The peak is added to the list,
the IBI plot and the CSV downloads. The plot is a small custom component in components/zoom_plot.
