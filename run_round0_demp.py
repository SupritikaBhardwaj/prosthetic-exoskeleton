"""
run_round0_demo.py

Round 0 demo:
- Recursively find CSV files (supports flat or subfolder-based dataset layouts)
- Load EMG CSVs (assumes rows=time samples, columns=channels OR one-channel vector)
- Preprocess: bandpass (20-450Hz), rectify, normalize
- Feature extraction per-sample: RMS, MAV, variance, ZCR (per channel -> concatenated)
- Train an SVM baseline and print metrics
- Simple live demo: randomly pick test samples, predict, print result and animate a virtual joint

Usage:
    python run_round0_demo.py

Dependencies:
    pip install numpy pandas scipy scikit-learn matplotlib joblib
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

# -------------------------------
# Configuration
# -------------------------------
FS_GUESS = 1000.0   # assumed sampling freq (Hz) if not provided in metadata
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 450.0
FILTER_ORDER = 4

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------------
# Utilities: load CSV files recursive
# -------------------------------
def find_csv_files(root="."):
    """Find CSV files recursively under root. Returns list of (filepath, label). 
    Label is inferred from the parent folder name; if not available uses filename prefix."""
    filepaths = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    pairs = []
    for fp in filepaths:
        # ignore common metadata files like README or requirements if they are csv by mistake
        base = os.path.basename(fp)
        if base.lower().startswith("readme") or base.lower().startswith("requirements"):
            continue
        parent = os.path.basename(os.path.dirname(fp))
        # If parent is '.' or root, fallback to filename-derived label
        if parent == "" or parent == os.path.basename(os.getcwd()):
            # try to infer label from filename tokens
            fname = os.path.splitext(base)[0]
            label = fname.split("_")[0].lower()
        else:
            label = parent.lower()
        pairs.append((fp, label))
    return pairs

# -------------------------------
# Signal processing
# -------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(sig, fs=FS_GUESS, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, order=FILTER_ORDER):
    # sig is 1D numpy array
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, sig)
    except Exception:
        # in case filtfilt fails on short signals, return original
        return sig

def rectify_and_envelope(sig, window_ms=50, fs=FS_GUESS):
    # full-wave rectification + moving RMS envelope
    rect = np.abs(sig)
    window_len = max(1, int((window_ms/1000.0) * fs))
    # simple moving average RMS-like envelope
    conv_kernel = np.ones(window_len) / window_len
    env = np.convolve(rect, conv_kernel, mode='same')
    return env

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features_single_signal(signal, fs=FS_GUESS):
    """Extract basic features for one 1D signal (array). Returns list of features."""
    # mean absolute value
    mav = np.mean(np.abs(signal))
    # RMS
    rms = np.sqrt(np.mean(signal**2))
    # variance
    var = np.var(signal)
    # zero crossing rate (approx)
    zcr = ((signal[:-1] * signal[1:]) < 0).sum()
    # waveform length
    wl = np.sum(np.abs(np.diff(signal)))
    return [mav, rms, var, zcr, wl]

def extract_features_multichannel(sample_array, fs=FS_GUESS):
    """
    sample_array: 2D np.array shape (n_samples, n_channels) OR (n_samples,) for single channel
    Returns concatenated features across channels.
    """
    sample = np.asarray(sample_array)
    if sample.ndim == 1:
        sample = sample.reshape(-1, 1)
    n_samples, n_channels = sample.shape
    feats = []
    for ch in range(n_channels):
        sig = sample[:, ch]
        env = rectify_and_envelope(sig, window_ms=50, fs=fs)
        feats += extract_features_single_signal(env, fs=fs)
    return np.array(feats)

# -------------------------------
# Data pipeline: load -> preprocess -> features
# -------------------------------
def load_csv_signal(path):
    """Load a CSV and return numpy array. Try to handle: single-column, multi-column, header/no-header."""
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=0)
    arr = df.values
    # If shape suggests channels in rows (channels x samples) we transpose to (samples, channels)
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 8:
        # Heuristic: if rows less than columns and rows small, might be channels in rows
        arr = arr.T
    return arr.astype(float)

def preprocess_and_extract_for_file(path, fs=FS_GUESS):
    arr = load_csv_signal(path)
    # filter each channel
    if arr.ndim == 1:
        filtered = bandpass_filter(arr, fs=fs)
    else:
        # arr shape (samples, channels)
        filtered = np.vstack([bandpass_filter(arr[:, ch], fs=fs) for ch in range(arr.shape[1])]).T
    # normalize per-channel (z-score)
    if filtered.ndim == 1:
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    else:
        mean = np.mean(filtered, axis=0)
        std = np.std(filtered, axis=0) + 1e-8
        filtered = (filtered - mean) / std
    feats = extract_features_multichannel(filtered, fs=fs)
    return feats

# -------------------------------
# Main: build dataset, train, demo
# -------------------------------
def build_dataset(root="."):
    pairs = find_csv_files(root)
    print(f"Found {len(pairs)} CSV files.")
    X = []
    y = []
    for fp, label in pairs:
        try:
            feats = preprocess_and_extract_for_file(fp)
            X.append(feats)
            y.append(label)
        except Exception as e:
            print(f"Skipping {fp} due to error: {e}")
    X = np.array(X)
    y = np.array(y)
    print("Features shape:", X.shape)
    return X, y, pairs

def train_and_save_model(X, y, save_path="model_round0.joblib"):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
    clf = SVC(kernel='rbf', probability=True, C=1.0, random_state=RANDOM_SEED)
    print("Training SVM (this may take a moment)...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy on holdout set: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump({"scaler": scaler, "model": clf}, save_path)
    print(f"Saved model + scaler to: {save_path}")
    return scaler, clf, (X_train, X_test, y_train, y_test)

# -------------------------------
# Simple animation demo mapping labels -> angles
# -------------------------------
def demo_live_prediction(model_bundle, file_label_pairs, demo_seconds=30, interval_s=1.0):
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]

    # prepare test list (use same files we loaded earlier)
    test_list = [fp for fp, label in file_label_pairs]

    # Matplotlib animation setup: single-joint limb from (0,0) to (cos(theta), sin(theta))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    line, = ax.plot([], [], lw=6, marker='o')
    text = ax.text(-1.0, 1.05, "", fontsize=12)

    def init():
        line.set_data([0, 0.5], [0, 0])
        text.set_text("")
        return line, text

    # Map labels to angles (radians) - adjust to your gestures
    unique_labels = list(set([lbl for _, lbl in file_label_pairs]))
    unique_labels_sorted = sorted(unique_labels)
    label_to_angle = {}
    span = np.pi/2  # 90 deg range
    for i, lbl in enumerate(unique_labels_sorted):
        label_to_angle[lbl] = -span/2 + i * (span / max(1, (len(unique_labels_sorted)-1)))
    print("Label -> angle mapping:", label_to_angle)

    start_time = time.time()
    last_update = 0

    def update(frame):
        nonlocal last_update
        now = time.time()
        if now - start_time > demo_seconds:
            plt.close(fig)
            return line, text

        # Only update every interval_s seconds
        if now - last_update < interval_s:
            return line, text
        last_update = now

        # pick a random test file, preprocess, predict
        fp = random.choice(test_list)
        try:
            feats = preprocess_and_extract_for_file(fp)
            feats_scaled = scaler.transform(feats.reshape(1, -1))
            probs = model.predict_proba(feats_scaled)[0]
            pred = model.predict(feats_scaled)[0]
            # confidence-like value
            conf = np.max(probs)
        except Exception as e:
            print("Prediction error for", fp, ":", e)
            return line, text

        angle = label_to_angle.get(pred, 0.0)
        x = [0, np.cos(angle)]
        y = [0, np.sin(angle)]
        line.set_data(x, y)
        text.set_text(f"Predicted: {pred}  (conf={conf:.2f})\nSource: {os.path.basename(fp)}")
        return line, text

    ani = animation.FuncAnimation(fig, update, init_func=init, interval=200, blit=False)
    plt.title("Round0: Virtual Joint responding to predicted intent")
    plt.show()


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    print("=== Round0 MVP pipeline ===")
    root = "."  # change if dataset is in specific folder like "emg_dataset"
    X, y, pairs = build_dataset(root)
    if len(X) == 0:
        print("No usable CSV files found under the current directory. Place your EMG CSVs under the project root or a subfolder and try again.")
        exit(1)

    scaler, model, splits = train_and_save_model(X, y, save_path="model_round0.joblib")
    # load back for demo
    bundle = joblib.load("model_round0.joblib")
    print("\nStarting live demo (30s). Close the animation window to end early.")
    demo_live_prediction(bundle, pairs, demo_seconds=30, interval_s=1.0)
    print("Demo finished.")
