from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="optimized_cnn_lstm_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

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

# --- Prediction Endpoint (POST from Form) ---
@app.post("/predict", response_class=HTMLResponse)
async def predict(ppg_sample: str = Form(...), fs_sensor: float = Form(...)):
    try:
        # Parse input string to list of floats
        signal = [float(x) for x in ppg_sample.strip().split(",") if x.strip()]
        
        # Normalize signal to match training range
        signal = np.array(signal, dtype=float)
        signal = (signal - signal.min()) / (signal.max() - signal.min())  # 0 to 1
        signal = signal * (2.0 + 1.5) - 1.5  # Scale to [-1.5, 2.0]

        # Pad or truncate to 875 samples
        if len(signal) < 875:
            signal = np.pad(signal, (0, 875 - len(signal)), mode='constant')
        else:
            signal = signal[:875]

        # Reshape to match model input shape
        model_input = signal.reshape(1, 875, 1).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details['index'], model_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]
        sbp, dbp = float(output[0]), float(output[1])

        return f"""
        <html>
            <body>
                <h2>Prediction Result</h2>
                <p><strong>SBP:</strong> {sbp:.2f} mmHg</p>
                <p><strong>DBP:</strong> {dbp:.2f} mmHg</p>
                <br><a href="/">Try another sample</a>
            </body>
        </html>
        """

    except Exception as e:
        return HTMLResponse(f"<h3>Error:</h3><pre>{str(e)}</pre>")

