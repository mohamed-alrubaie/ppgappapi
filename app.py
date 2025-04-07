# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
import tensorflow as tf

app = FastAPI()

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# --- Known MinMax scaler parameters ---
SBP_MIN, SBP_RANGE = 55.0, 144.0   # 199 - 55
DBP_MIN, DBP_RANGE = 40.0, 78.0    # 118 - 40

def inverse_sbp(scaled_vals):
    return [v * SBP_RANGE + SBP_MIN for v in scaled_vals]

def inverse_dbp(scaled_vals):
    return [v * DBP_RANGE + DBP_MIN for v in scaled_vals]

# --- Request schema ---
class SensorData(BaseModel):
    sensor_signal: list[float]

# --- Helper functions ---
def normalize_to_range(x, lower=-1.5, upper=2.0):
    x = np.array(x, dtype=float)
    return lower + (x - x.min()) * (upper - lower) / (x.max() - x.min())

def resample_signal(x, fs_orig=25, fs_target=125):
    n = int(len(x) * fs_target / fs_orig)
    return signal.resample(x, n)

def segment_samples(ppg, fs=125):
    length = fs * 7
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
    return systolic + peaks[-1] if len(peaks) > 0 else systolic

def detect_peaks_custom(ppg, fs=125):
    sys_p, _ = find_peaks(ppg, distance=fs/2.5)
    dia_p = [
        np.argmin(ppg[sys_p[i-1]:sys_p[i]]) + sys_p[i-1]
        for i in range(1, len(sys_p))
    ]
    notches = []
    for i in range(1, len(sys_p)):
        s, e = sys_p[i-1], dia_p[i-1] if i-1 < len(dia_p) else len(ppg)
        notch = dicrotic_notch(ppg[s:e], 0)
        if notch:
            notches.append(s + notch)
    return sys_p, dia_p, notches

def extract_features(seg, sys_p, dia_p, notches, fs=125):
    feats = []
    # 1. Peak-to-peak & HR
    if len(sys_p) > 1:
        iv = np.diff(sys_p)/fs
        feats += [iv.mean(), 60/iv.mean()]
    else:
        feats += [0,0]
    # 2. Slopes
    if len(sys_p)>0:
        rise = [(seg[i]-seg[i-1])/(1/fs) for i in sys_p if i>0]
        feats.append(np.mean(rise) if len(rise)>0 else 0)
        fall = [
            (seg[sys_p[j]]-seg[sys_p[j+1]])/((sys_p[j+1]-sys_p[j])/fs)
            for j in range(len(sys_p)-1)
        ]
        feats.append(np.mean(fall) if len(fall)>0 else 0)
    else:
        feats += [0,0]
    # 3. Down/upstroke
    if len(dia_p)>0 and len(notches)>0:
        feats += [(dia_p[0]-notches[0])/fs,
                  (notches[0]-sys_p[0])/fs if len(sys_p)>0 else 0]
    else:
        feats += [0,0]
    # 4. Percentiles
    for p in [10,25,33,50,66,75,100]:
        if len(sys_p)>0 and len(dia_p)>0:
            feats.append(np.percentile(seg[sys_p[0]:dia_p[0]], p))
        else:
            feats.append(0)
        if len(dia_p)>0 and len(sys_p)>1:
            feats.append(np.percentile(seg[dia_p[0]:sys_p[1]], p))
        else:
            feats.append(0)
    # 5. Amplitude
    feats += [seg.max(), seg.min()]
    # 6. Frequency
    N = len(seg)
    freqs = np.fft.fftfreq(N,1/fs)
    fv    = np.abs(fft(seg))
    dom   = np.argmax(fv[1:])+1
    feats.append(freqs[dom])
    half  = fv.max()/2
    bw    = freqs[fv>=half]
    feats.append((bw.max()-bw.min()) if len(bw)>0 else 0)
    # 7. Symmetry
    if len(sys_p)>0 and len(dia_p)>0:
        asc  = dia_p[0]-sys_p[0]
        desc = (sys_p[1]-dia_p[0]) if len(sys_p)>1 else 0
        feats.append(asc/(desc+1e-6))
    else:
        feats.append(0)
    # 8. Hysteresis, deriv, curvature, lag
    feats.append(np.trapz(np.abs(np.diff(seg))))
    feats.append(np.mean(np.abs(np.diff(seg))))
    feats.append(np.sum(np.abs(np.diff(np.diff(seg))))/N)
    feats.append(np.mean(np.diff(sys_p)) if len(sys_p)>1 else 0)
    return np.array(feats, dtype=np.float32)

@app.post("/predict")
async def predict(data: SensorData):
    try:
        # 1. Normalize (assuming original fs=25 Hz)
        normed    = normalize_to_range(data.sensor_signal)
        # 2. Resample to 125 Hz
        resampled = resample_signal(normed, fs_orig=25, fs_target=125)
        # 3. Segment into 7s windows
        segments  = segment_samples(resampled)
        # 4. Build feature matrix
        feat_list = []
        for seg in segments:
            sys_p, dia_p, notches = detect_peaks_custom(seg)
            feat_list.append(extract_features(seg, sys_p, dia_p, notches))
        feat_matrix = np.vstack(feat_list)
        # 5. Pad & reshape
        ts = 2
        nf = feat_matrix.shape[1]
        if nf % ts:
            p = ts - (nf % ts)
            feat_matrix = np.pad(feat_matrix, ((0,0),(0,p)), 'constant')
            nf += p
        fps = nf // ts
        model_input = feat_matrix.reshape(-1, ts, fps).astype(np.float32)
        # 6. Inference
        sbp_scaled = []
        dbp_scaled = []
        for sample in model_input:
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])[0]
            sbp_scaled.append(float(out[0]))
            dbp_scaled.append(float(out[1]))
        # 7. Inverse scale
        sbp = inverse_sbp(sbp_scaled)
        dbp = inverse_dbp(dbp_scaled)
        # 8. Return plain Python lists
        return {"SBP": sbp, "DBP": dbp}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
