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
    derivative = np.diff(np.diff(beat[systolic:]))  # Compute second derivative
    point = find_peaks(derivative)[0]                # Locate peaks in the derivative
    corrected = 0
    if len(point) > 0:
        corrected = systolic + point[-1]             # Adjust position back to original beat
    return corrected

def detect_peaks_custom(ppg_signal, fs=125):
    # 1. Systolic peaks
    systolic_peaks, _ = find_peaks(ppg_signal, distance=fs/2.5)
    
    # 2. Diastolic peaks: the minima between successive systolic peaks
    diastolic_peaks = []
    for i in range(1, len(systolic_peaks)):
        seg = ppg_signal[systolic_peaks[i-1]:systolic_peaks[i]]
        diastolic_peak = np.argmin(seg) + systolic_peaks[i-1]
        diastolic_peaks.append(diastolic_peak)
    
    # 3. Dicrotic notches: second-derivative peaks after each systolic
    dicrotic_notches = []
    for i in range(1, len(systolic_peaks)):
        start_idx = systolic_peaks[i-1]
        end_idx   = diastolic_peaks[i-1] if i-1 < len(diastolic_peaks) else len(ppg_signal)
        segment   = ppg_signal[start_idx:end_idx]
        if len(segment) > 10:
            notch = dicrotic_notch(segment, 0)
            if notch:
                dicrotic_notches.append(start_idx + notch)
    
    return systolic_peaks, diastolic_peaks, dicrotic_notches

def segment_samples(ppg_signal, fs=125):
    sample_length = fs * 7  # 7‑second windows
    num_samples   = len(ppg_signal) // sample_length
    segments      = []
    
    for i in range(num_samples):
        start   = i * sample_length
        end     = start + sample_length
        segments.append(ppg_signal[start:end])
    
    return segments

def extract_features(segment, systolic_peaks, diastolic_peaks, fs=125):
    features = []
    
    # Peak‑to‑peak intervals & heart rate
    if len(systolic_peaks) > 1:
        intervals = np.diff(systolic_peaks) / fs
        features.append(np.mean(intervals))
        features.append(60 / np.mean(intervals))
    else:
        features.extend([0, 0])
    
    # Blood‑flow acceleration & deceleration
    if len(systolic_peaks) > 0:
        # Rising slope
        rise = []
        for idx in systolic_peaks:
            if idx > 0:
                rise.append((segment[idx] - segment[idx-1]) / (1/fs))
        features.append(np.mean(rise) if rise else 0)
        # Falling slope
        fall = []
        for j in range(len(systolic_peaks)-1):
            curr, nxt = systolic_peaks[j], systolic_peaks[j+1]
            fall.append((segment[curr] - segment[nxt]) / ((nxt-curr)/fs))
        features.append(np.mean(fall) if fall else 0)
    else:
        features.extend([0, 0])
    
    # Downstroke & upstroke times
    if diastolic_peaks and dicrotic_notches:
        features.append((diastolic_peaks[0] - dicrotic_notches[0]) / fs)
        features.append((dicrotic_notches[0] - systolic_peaks[0]) / fs if systolic_peaks else 0)
    else:
        features.extend([0, 0])
    
    # Percentile areas between peaks
    percentiles = [10, 25, 33, 50, 66, 75, 100]
    for p in percentiles:
        if systolic_peaks and diastolic_peaks:
            s, d = systolic_peaks[0], diastolic_peaks[0]
            features.append(np.percentile(segment[s:d], p))
        else:
            features.append(0)
        if diastolic_peaks and len(systolic_peaks)>1:
            d, s2 = diastolic_peaks[0], systolic_peaks[1]
            features.append(np.percentile(segment[d:s2], p))
        else:
            features.append(0)
    
    # Amplitude stats
    features.append(np.max(segment))
    features.append(np.min(segment))
    
    # Frequency‑domain: dominant freq & bandwidth
    N = len(segment)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_vals = np.abs(fft(segment))
    dom = np.argmax(fft_vals[1:]) + 1
    features.append(freqs[dom])
    half_max = fft_vals.max()/2
    bw = freqs[np.where(fft_vals>=half_max)]
    features.append((bw.max()-bw.min()) if len(bw) else 0)
    
    # Beat symmetry
    if systolic_peaks and diastolic_peaks:
        asc = diastolic_peaks[0] - systolic_peaks[0]
        desc = (systolic_peaks[1] - diastolic_peaks[0]) if len(systolic_peaks)>1 else 0
        features.append(asc/(desc+1e-6))
    else:
        features.append(0)
    
    # Hysteresis, derivative, curvature, lag
    features.append(np.trapz(np.abs(np.diff(segment))))
    features.append(np.mean(np.abs(np.diff(segment))))
    features.append(np.sum(np.abs(np.diff(np.diff(segment))))/N)
    if len(systolic_peaks)>1:
        features.append(np.mean(np.diff(systolic_peaks)))
    else:
        features.append(0)
    
    return np.array(features, dtype=np.float32)
# ---- Prediction endpoint ----

@app.post("/predict")
def predict(data: SensorData):
    try:
        # 1. Normalize raw signal to the training range
        normalized = normalize_to_range(data.sensor_signal, lower=-1.5, upper=2.0)

        # 2. Resample from original sensor rate to 125 Hz
        resampled = resample_signal(normalized, fs_orig=data.fs_sensor, fs_target=125)

        # 3. Segment into 7‑second windows (padding the last if needed)
        segments = segment_samples(resampled, fs=125)

        # 4. Extract features for each segment
        feature_list = []
        for seg in segments:
            sys_peaks, dia_peaks, notches = detect_peaks_custom(seg, fs=125)
            feats = extract_features(seg, sys_peaks, dia_peaks, fs=125)
            feature_list.append(feats)
        feat_matrix = np.vstack(feature_list)  # shape: (n_segments, n_features)

        # 5. Pad feature dimension so it's divisible by time_steps, then reshape
        time_steps = 2
        n_feats = feat_matrix.shape[1]
        if n_feats % time_steps:
            pad = time_steps - (n_feats % time_steps)
            feat_matrix = np.pad(feat_matrix, ((0, 0), (0, pad)), 'constant')
            n_feats += pad
        features_per_step = n_feats // time_steps
        model_input = feat_matrix.reshape(-1, time_steps, features_per_step).astype(np.float32)

        # 6. Run inference on each time‑windowed sample
        sbp_preds = []
        dbp_preds = []
        for sample in model_input:
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])[0]
            sbp_preds.append(float(output[0]))
            dbp_preds.append(float(output[1]))

        # 7. Return the predictions
        return {"SBP": sbp_preds, "DBP": dbp_preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
