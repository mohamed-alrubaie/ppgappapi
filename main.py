# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
import tensorflow as tf

# ---- Load TFLite model at startup ----

interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# ---- FastAPI setup ----

app = FastAPI()

class SensorData(BaseModel):
    sensor_signal: list[float]
    fs_sensor: float  # original sampling rate of the sensor

# ---- Helper functions ----

def normalize_to_range(x, lower=-1.5, upper=2.0):
    x = np.array(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    return lower + (x - x_min) * (upper - lower) / (x_max - x_min)

def resample_signal(x, fs_orig, fs_target=125):
    num_samples = int(len(x) * fs_target / fs_orig)
    return signal.resample(x, num_samples)

def segment_samples(ppg, fs=125):
    length = int(fs * 7)  # 7‑second windows
    segments = []
    for start in range(0, len(ppg), length):
        seg = ppg[start:start+length]
        if len(seg) < length:
            seg = np.pad(seg, (0, length - len(seg)), 'constant')
        segments.append(seg)
    return segments

def dicrotic_notch(beat, systolic):
    deriv2 = np.diff(np.diff(beat[systolic:]))
    peaks, _ = find_peaks(deriv2)
    return systolic + peaks[-1] if len(peaks) else systolic

def detect_peaks_custom(ppg, fs=125):
    sys_peaks, _ = find_peaks(ppg, distance=fs/2.5)
    dias_peaks = [
        np.argmin(ppg[sys_peaks[i-1]:sys_peaks[i]]) + sys_peaks[i-1]
        for i in range(1, len(sys_peaks))
    ]
    notches = []
    for i in range(1, len(sys_peaks)):
        start, end = sys_peaks[i-1], dias_peaks[i-1] if i-1 < len(dias_peaks) else len(ppg)
        notches.append(dicrotic_notch(ppg[start:end], 0) + start)
    return sys_peaks, dias_peaks, notches

def extract_features(seg, sys_peaks, dias_peaks, fs=125):
    feats = []
    # Example features: mean interval & heart rate
    if len(sys_peaks)>1:
        intervals = np.diff(sys_peaks)/fs
        feats.append(intervals.mean())
        feats.append(60.0/intervals.mean())
    else:
        feats += [0.0, 0.0]
    # Add more features exactly as your model expects...
    # Here we pad/truncate to the feature length your model was trained on:
    TARGET_FEATURE_COUNT = 36  # <-- adjust to your real feature count
    if len(feats) < TARGET_FEATURE_COUNT:
        feats += [0.0] * (TARGET_FEATURE_COUNT - len(feats))
    else:
        feats = feats[:TARGET_FEATURE_COUNT]
    return np.array(feats, dtype=np.float32)

# ---- Prediction endpoint ----

@app.post("/predict")
def predict(data: SensorData):
    try:
        # 1. Normalize
        normalized = normalize_to_range(data.sensor_signal, -1.5, 2.0)
        # 2. Resample to 125 Hz
        resampled = resample_signal(normalized, data.fs_sensor, fs_target=125)
        # 3. Segment into 7 s windows
        segments = segment_samples(resampled, fs=125)
        # 4. Extract features per segment
        feat_matrix = np.vstack([
            extract_features(
                seg,
                *detect_peaks_custom(seg, fs=125),
                fs=125
            )
            for seg in segments
        ])  # shape: (n_segments, TARGET_FEATURE_COUNT)
        # 5. Pad & reshape for time_steps=2
        time_steps = 2
        n_feats = feat_matrix.shape[1]
        if n_feats % time_steps:
            pad = time_steps - (n_feats % time_steps)
            feat_matrix = np.pad(feat_matrix, ((0,0),(0,pad)), 'constant')
            n_feats += pad
        fps = n_feats // time_steps
        inp = feat_matrix.reshape(-1, time_steps, fps).astype(np.float32)
        # 6. TFLite inference
        sbp_preds, dbp_preds = [], []
        for sample in inp:
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])[0]
            sbp_preds.append(float(out[0]))
            dbp_preds.append(float(out[1]))
        # 7. Return raw predictions
        return {"SBP": sbp_preds, "DBP": dbp_preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
