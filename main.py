# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.fft import fft

# ---- Load TFLite model at startup ----

interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# ---- FastAPI setup ----

app = FastAPI()

class SensorData(BaseModel):
    sensor_signal: list[float]

# ---- Helper functions ----

def segment_samples(ppg_signal, fs=125):
    sample_length = fs * 7
    num_segments = len(ppg_signal) // sample_length
    return [ppg_signal[i*sample_length:(i+1)*sample_length]
            for i in range(num_segments)]

def dicrotic_notch(beat, systolic):
    deriv2 = np.diff(np.diff(beat[systolic:]))
    peaks, _ = find_peaks(deriv2)
    return systolic + peaks[-1] if len(peaks) else systolic

def detect_peaks_custom(ppg_signal, fs=125):
    # systolic peaks
    systolic_peaks, _ = find_peaks(ppg_signal, distance=fs/2.5)
    # diastolic = minimum between successive systolic
    diastolic_peaks = [
        np.argmin(ppg_signal[systolic_peaks[i-1]:systolic_peaks[i]]) + systolic_peaks[i-1]
        for i in range(1, len(systolic_peaks))
    ]
    # dicrotic notch
    dicrotic_notches = []
    for i in range(1, len(systolic_peaks)):
        start = systolic_peaks[i-1]
        end = diastolic_peaks[i-1] if i-1 < len(diastolic_peaks) else len(ppg_signal)
        notch = dicrotic_notch(ppg_signal[start:end], 0)
        dicrotic_notches.append(start + notch)
    return systolic_peaks, diastolic_peaks, dicrotic_notches

def extract_features(segment, systolic_peaks, diastolic_peaks, fs=125):
    feats = []
    # Example: mean peak interval and heart rate
    if len(systolic_peaks)>1:
        intervals = np.diff(systolic_peaks)/fs
        feats.append(np.mean(intervals))
        feats.append(60/np.mean(intervals))
    else:
        feats += [0,0]
    # Add more features here exactly as you trained your model...
    # For brevity, let’s pad to a fixed length:
    # Suppose your model was trained on 20 features per segment:
    feats = feats + [0]*(20-len(feats))
    return np.array(feats, dtype=np.float32)

# ---- Prediction endpoint ----

@app.post("/predict")
def predict(data: SensorData):
    try:
        # 1. Segment raw signal
        segments = segment_samples(data.sensor_signal)
        # 2. Extract features
        feats = np.vstack([
            extract_features(
                seg,
                *detect_peaks_custom(seg, fs=125),
                fs=125
            )
            for seg in segments
        ])  # shape (n_segments, n_features)

        # 3. Pad & reshape for time_steps=2
        time_steps = 2
        n_feats = feats.shape[1]
        if n_feats % time_steps:
            pad = time_steps - (n_feats % time_steps)
            feats = np.pad(feats, ((0,0),(0,pad)), 'constant')
            n_feats += pad
        fps = n_feats // time_steps
        inp = feats.reshape(-1, time_steps, fps).astype(np.float32)

        # 4. Run TFLite inference
        sbp_preds, dbp_preds = [], []
        for sample in inp:
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])[0]
            sbp_preds.append(float(out[0]))
            dbp_preds.append(float(out[1]))

        return {"SBP": sbp_preds, "DBP": dbp_preds}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
