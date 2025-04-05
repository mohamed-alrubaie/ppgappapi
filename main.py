# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft
# ---- Load artifacts at startup ----

# 1. Load scalers
with open('scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('sbp_scaler.pkl', 'rb') as f:
    sbp_scaler = pickle.load(f)
with open('dbp_scaler.pkl', 'rb') as f:
    dbp_scaler = pickle.load(f)

# 2. Load TFLite model
interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# ---- FastAPI setup ----

app = FastAPI()

class SensorData(BaseModel):
    sensor_signal: list[float]

# ---- Helper functions (you must implement these) ----

def segment_samples(ppg_signal, fs=125):
    sample_length = fs * 7
    num_segments = len(ppg_signal) // sample_length
    return [ppg_signal[i*sample_length:(i+1)*sample_length]
            for i in range(num_segments)]


def dicrotic_notch(beat, systolic):
    derivative = np.diff(np.diff(beat[systolic:]))  # Compute second derivative
    point = find_peaks(derivative)[0]  # Locate peaks in the derivative
    corrected = 0
    if len(point) > 0:
        corrected = systolic + point[-1]  # Adjust the position to the original beat
    return corrected

def detect_peaks_custom(ppg_signal, fs=125):
    systolic_peaks, _ = find_peaks(ppg_signal, distance=fs/2.5)  # Identify systolic peaks
    
    diastolic_peaks = []
    for i in range(1, len(systolic_peaks)):
        segment = ppg_signal[systolic_peaks[i-1]:systolic_peaks[i]]
        diastolic_peak = np.argmin(segment) + systolic_peaks[i-1]  # Find diastolic (minimum) point
        diastolic_peaks.append(diastolic_peak)
    
    dicrotic_notches = []
    for i in range(1, len(systolic_peaks)):
        start_idx = systolic_peaks[i-1]
        end_idx = diastolic_peaks[i-1] if i <= len(diastolic_peaks) else len(ppg_signal)
        segment = ppg_signal[start_idx:end_idx]
        if len(segment) > 10:
            notch_idx = dicrotic_notch(segment, 0)
            if notch_idx:
                dicrotic_notches.append(start_idx + notch_idx)
    
    return systolic_peaks, diastolic_peaks, dicrotic_notches

def segment_samples(ppg_signal, fs=125):
    sample_length = fs * 7  # 7-second segments
    num_samples = len(ppg_signal) // sample_length
    segments = []
    
    for i in range(num_samples):
        start = i * sample_length
        end = start + sample_length
        segment = ppg_signal[start:end]
        segments.append(segment)
    
    return segments

def extract_features(segment, systolic_peaks, diastolic_peaks, fs=125):
    features = []
    
    
    
    # Peak-to-peak intervals
    if len(systolic_peaks) > 1:
        peak_intervals = np.diff(systolic_peaks) / fs
        features.append(np.mean(peak_intervals))
        
        features.append(60 / np.mean(peak_intervals))
    else:
        features.extend([0, 0, 0])
   # Blood flow acceleration and deceleration rates
    if len(systolic_peaks) > 0:
        # Rising slope (acceleration)
        rise_slope = []
        for i in range(len(systolic_peaks)):
            if systolic_peaks[i] > 0:
                start_idx = max(0, systolic_peaks[i] - 1)
                rise_slope.append((segment[systolic_peaks[i]] - segment[start_idx]) / (1 / fs))
        features.append(np.mean(rise_slope) if rise_slope else 0)

        # Falling slope (deceleration)
        fall_slope = []
        for i in range(len(systolic_peaks) - 1):
            end_idx = systolic_peaks[i + 1]
            fall_slope.append((segment[systolic_peaks[i]] - segment[end_idx]) / ((end_idx - systolic_peaks[i]) / fs))
        features.append(np.mean(fall_slope) if fall_slope else 0)
    else:
        features.extend([0, 0])  # Fallback values if no peaks are detected  
    # downstroke time
    if len(diastolic_peaks) > 0 and len(dicrotic_notches) > 0:
        diastolic_to_notch_time = (diastolic_peaks[0] - dicrotic_notches[0]) / fs
        features.append(diastolic_to_notch_time)
    else:
        features.append(0)  # Fallback value


    # Upstroke time
    if len(systolic_peaks) > 0 and len(dicrotic_notches) > 0:
        upstroke_time = (dicrotic_notches[0] - systolic_peaks[0]) / fs
        features.append(upstroke_time)
    else:
        features.append(0)
    
    # Percentile areas
    percentiles = [10, 25, 33, 50, 66, 75, 100]
    for percentile in percentiles:
        if len(systolic_peaks) > 0 and len(diastolic_peaks) > 0:
            s_d_area = np.percentile(segment[systolic_peaks[0]:diastolic_peaks[0]], percentile)
        else:
            s_d_area = 0
        features.append(s_d_area)
        
        if len(diastolic_peaks) > 0 and len(systolic_peaks) > 1:
            d_s_area = np.percentile(segment[diastolic_peaks[0]:systolic_peaks[1]], percentile)
        else:
            d_s_area = 0
        features.append(d_s_area)
    
    # Max, Min, and Std Amplitude
    features.append(np.max(segment))
    features.append(np.min(segment))
   
    
    # Frequency-domain features
    N = len(segment)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_values = np.abs(fft(segment))
    dominant_freq_idx = np.argmax(fft_values[1:]) + 1
    features.append(freqs[dominant_freq_idx])

       

    # Bandwidth
    half_max_power = np.max(fft_values) / 2
    bandwidth_freqs = freqs[np.where(fft_values >= half_max_power)]
    if len(bandwidth_freqs) > 0:
        bandwidth = np.max(bandwidth_freqs) - np.min(bandwidth_freqs)
    else:
        bandwidth = 0
    features.append(bandwidth)

    # New Feature: Beat Symmetry
    
    if len(systolic_peaks) > 0 and len(diastolic_peaks) > 0:
        ascending_time = diastolic_peaks[0] - systolic_peaks[0]
        descending_time = systolic_peaks[1] - diastolic_peaks[0] if len(systolic_peaks) > 1 else 0
        symmetry_ratio = ascending_time / (descending_time + 1e-6)  # Avoid division by zero
        features.append(symmetry_ratio) 
    else:
        features.append(0) 
            # Hysteresis
    hysteresis = np.trapz(np.abs(np.diff(segment)))
    features.append(hysteresis)

    # Waveform Derivatives
    waveform_derivative = np.mean(np.abs(np.diff(segment)))
    features.append(waveform_derivative)



    # PPG Beat Curvature
    curvature = np.sum(np.abs(np.diff(np.diff(segment)))) / len(segment)
    features.append(curvature)


    # Lag Between Consecutive Peaks
    if len(systolic_peaks) > 1:
        lags = np.diff(systolic_peaks)
        features.append(np.mean(lags))

    else:
        features.extend([0, 0])


    return np.array(features)
filtered_ppg_signal = filtered_ppg_data[0]
segments = segment_samples(filtered_ppg_signal)

all_features = []
for segment in segments:
    systolic_peaks, diastolic_peaks, dicrotic_notches = detect_peaks_custom(segment)
    features = extract_features(segment, systolic_peaks, diastolic_peaks)
    all_features.append(features)

all_features = np.array(all_features)
print(f"Extracted features shape: {all_features.shape}")
print(f"Sample features: {all_features[0]}")


# ---- Prediction endpoint ----

@app.post("/predict")
def predict(data: SensorData):
    try:
        # 1. Segment
        segments = segment_samples(data.sensor_signal)
        # 2. Extract features for each segment
        feats = []
        for seg in segments:
            sp, dp = detect_peaks_custom(seg, fs=125)
            f = extract_features(seg, sp, dp, fs=125)
            feats.append(f)
        feats = np.vstack(feats)  # shape (n_segments, n_features)

        # 3. Scale
        feats_scaled = feature_scaler.transform(feats)

        # 4. Pad & reshape for time_steps=2
        time_steps = 2
        n_feats = feats_scaled.shape[1]
        if n_feats % time_steps:
            pad = time_steps - (n_feats % time_steps)
            feats_scaled = np.pad(feats_scaled, ((0,0),(0,pad)), 'constant')
            n_feats += pad
        fps = n_feats // time_steps
        input_data = feats_scaled.reshape(-1, time_steps, fps).astype(np.float32)

        # 5. Run TFLite inference
        sbp_preds = []
        dbp_preds = []
        for sample in input_data:
            # set input
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])[0]
            sbp_preds.append(out[0])
            dbp_preds.append(out[1])

        # 6. Inverse scale
        sbp = sbp_scaler.inverse_transform(np.array(sbp_preds).reshape(-1,1)).flatten().tolist()
        dbp = dbp_scaler.inverse_transform(np.array(dbp_preds).reshape(-1,1)).flatten().tolist()

        return {"SBP": sbp, "DBP": dbp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
