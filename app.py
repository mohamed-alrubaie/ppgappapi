from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from scipy import signal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft
import tensorflow as tf
app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
# Known scaler parameters
SBP_MIN   = 55.0
SBP_RANGE = 144.0  # 199 - 55

DBP_MIN   = 40.0
DBP_RANGE = 78.0   # 118 - 40

def inverse_sbp(scaled_vals):
    # scaled_vals: array‑like of floats in [0,1]
    return [v * SBP_RANGE + SBP_MIN for v in scaled_vals]

def inverse_dbp(scaled_vals):
    return [v * DBP_RANGE + DBP_MIN for v in scaled_vals]


# --- Homepage with Form ---
@app.get("/", response_class=HTMLResponse)
async def form_get():
    return """
    <html>
        <head><title>PPG Blood Pressure Predictor</title></head>
        <body>
            <h2>Enter PPG Input Sample</h2>
            <form action="/predict" method="post">
                <textarea name="ppg_sample" rows="20" cols="100"></textarea><br><br>
                <input type="text" name="fs_sensor" value="125" placeholder="Sampling rate (Hz)"><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """
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
        end_idx   = diastolic_peaks[i-1] if (i-1) < len(diastolic_peaks) else len(ppg_signal)
        segment   = ppg_signal[start_idx:end_idx]
        if len(segment) > 10:
            notch = dicrotic_notch(segment, 0)
            if notch != 0:
                dicrotic_notches.append(start_idx + notch)
    
    return systolic_peaks, diastolic_peaks, dicrotic_notches

def segment_samples(ppg_signal, fs=125):
    sample_length = fs * 7  # 7‑second windows
    num_samples   = len(ppg_signal) // sample_length
    segments      = []
    
    for i in range(num_samples):
        start = i * sample_length
        end   = start + sample_length
        segments.append(ppg_signal[start:end])
    
    return segments

def extract_features(segment, systolic_peaks, diastolic_peaks, dicrotic_notches, fs=125):
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
        features.append(np.mean(rise) if len(rise) > 0 else 0)
        # Falling slope
        fall = []
        for j in range(len(systolic_peaks)-1):
            curr, nxt = systolic_peaks[j], systolic_peaks[j+1]
            fall.append((segment[curr] - segment[nxt]) / ((nxt-curr)/fs))
        features.append(np.mean(fall) if len(fall) > 0 else 0)
    else:
        features.extend([0, 0])
    
    # Downstroke & upstroke times
    if len(diastolic_peaks) > 0 and len(dicrotic_notches) > 0:
        features.append((diastolic_peaks[0] - dicrotic_notches[0]) / fs)
        features.append((dicrotic_notches[0] - systolic_peaks[0]) / fs if len(systolic_peaks) > 0 else 0)
    else:
        features.extend([0, 0])
    
    # Percentile areas between peaks
    percentiles = [10, 25, 33, 50, 66, 75, 100]
    for p in percentiles:
        if len(systolic_peaks) > 0 and len(diastolic_peaks) > 0:
            s, d = systolic_peaks[0], diastolic_peaks[0]
            features.append(np.percentile(segment[s:d], p))
        else:
            features.append(0)
        if len(diastolic_peaks) > 0 and len(systolic_peaks) > 1:
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
    half_max = fft_vals.max() / 2
    bw = freqs[np.where(fft_vals >= half_max)]
    features.append((bw.max() - bw.min()) if len(bw) > 0 else 0)
    
    # Beat symmetry
    if len(systolic_peaks) > 0 and len(diastolic_peaks) > 0:
        asc = diastolic_peaks[0] - systolic_peaks[0]
        desc = (systolic_peaks[1] - diastolic_peaks[0]) if len(systolic_peaks) > 1 else 0
        features.append(asc / (desc + 1e-6))
    else:
        features.append(0)
    
    # Hysteresis, derivative, curvature, lag
    features.append(np.trapz(np.abs(np.diff(segment))))
    features.append(np.mean(np.abs(np.diff(segment))))
    features.append(np.sum(np.abs(np.diff(np.diff(segment)))) / N)
    if len(systolic_peaks) > 1:
        features.append(np.mean(np.diff(systolic_peaks)))
    else:
        features.append(0)
    
    return np.array(features, dtype=np.float32)

@app.post("/predict", response_class=HTMLResponse)
async def predict(ppg_sample: str = Form(...), fs_sensor: float = Form(...)):
    try:
        raw_signal = [float(x) for x in ppg_sample.strip().split(",") if x.strip()]

        # 1. Normalize and resample as before…
        normalized = normalize_to_range(raw_signal, -1.5, 2.0)
        resampled = resample_signal(normalized, fs_orig=fs_sensor, fs_target=125)
        segments  = segment_samples(resampled, fs=125)

        # 2. Feature extraction…
        feature_list = []
        for seg in segments:
            sys_p, dia_p, notches = detect_peaks_custom(seg, fs=125)
            feature_list.append(extract_features(seg, sys_p, dia_p, notches, fs=125))
        feat_matrix = np.vstack(feature_list)

        # 3. Pad & reshape for model…
        time_steps = 2
        n_feats    = feat_matrix.shape[1]
        if n_feats % time_steps:
            pad = time_steps - (n_feats % time_steps)
            feat_matrix = np.pad(feat_matrix, ((0,0),(0,pad)), 'constant')
            n_feats += pad
        fps         = n_feats // time_steps
        model_input = feat_matrix.reshape(-1, time_steps, fps).astype(np.float32)

        # 4. Run inference
        sbp_preds, dbp_preds = [], []
        for sample in model_input:
            interpreter.set_tensor(input_details['index'], [sample])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details['index'])[0]
            sbp_preds.append(out[0])
            dbp_preds.append(out[1])

        # 5. Reverse scaling
        sbp_orig = inverse_sbp(sbp_preds)
        dbp_orig = inverse_dbp(dbp_preds)

        # 6. Display first segment’s BP
        sbp = sbp_orig[0]
        dbp = dbp_orig[0]

        return f"""
        <html><body>
          <h2>Prediction Result</h2>
          <p><strong>SBP:</strong> {sbp:.2f} mmHg</p>
          <p><strong>DBP:</strong> {dbp:.2f} mmHg</p>
          <br><a href="/">Try another sample</a>
        </body></html>
        """

    except Exception as e:
        return HTMLResponse(f"<h3>Error:</h3><pre>{str(e)}</pre>")
