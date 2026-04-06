import gradio as gr
import cv2
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, detrend
import time

# --- Config & Signal Processing ---
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FS = 15  # Assumed Frames Per Second
WINDOW_SIZE = FS * 10  # 10-second sliding window

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=15, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_hr(buffer):
    if len(buffer) < FS * 3: return 0, 0
    
    # 1. Detrend and Filter
    signal = detrend(np.array(buffer))
    filtered = bandpass_filter(signal)
    
    # 2. FFT for Peak Frequency
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1.0/FS)
    
    # Focus only on human HR range (42-240 BPM)
    valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
    if len(valid_idx) == 0: return 0, 0
    
    peak_idx = valid_idx[np.argmax(fft[valid_idx])]
    bpm = freqs[peak_idx] * 60
    
    # 3. Quality Indicator (SNR-like)
    snr = fft[peak_idx] / np.mean(fft[valid_idx])
    return int(bpm), snr

# --- Main Logic ---
def process_vitals(image, state):
    if image is None: return None, state, ""
    
    # Initialize state if empty
    if state is None:
        state = {"buffer": [], "start_time": time.time(), "history": []}
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    display_text = "⚠️ No Face Detected"
    hr_val, quality = 0, 0
    
    for (x, y, w, h) in faces:
        # ROI: Forehead region (approx 20% of face height)
        roi_y1, roi_y2 = y + int(h*0.1), y + int(h*0.25)
        roi_x1, roi_x2 = x + int(w*0.3), x + int(w*0.7)
        
        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size > 0:
            # Extract mean Green channel
            green_mean = np.mean(roi[:, :, 1])
            state["buffer"].append(green_mean)
            if len(state["buffer"]) > WINDOW_SIZE: state["buffer"].pop(0)
            
            hr_val, quality = calculate_hr(state["buffer"])
            if hr_val > 0: state["history"].append(hr_val)
            
        # Draw UI Overlays
        color = (0, 255, 0) if quality > 2.5 else (0, 255, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)
        
        quality_str = "High" if quality > 3 else "Medium" if quality > 1.5 else "Low"
        display_text = f"💓 HR: {hr_val if hr_val > 0 else '--'} BPM | Signal: {quality_str}"
        cv2.putText(img, display_text, (x, y-10), 1, 1.5, color, 2)

    elapsed = int(time.time() - state["start_time"])
    summary = f"⏱️ Measuring... {elapsed}s\n{display_text}"
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), state, summary

def show_results(state):
    if not state or not state["history"]:
        return "No data collected. Try staying still for 10+ seconds."
    
    avg_hr = int(np.median(state["history"]))
    duration = int(time.time() - state["start_time"])
    stability = "Stable" if np.std(state["history"]) < 5 else "Variable"
    
    return f"""
    ### 📊 Measurement Results
    - **Average Heart Rate:** {avg_hr} BPM
    - **Duration:** {duration} seconds
    - **Confidence:** {stability}
    """

# --- Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State()
    
    gr.Markdown("# 🫀 AI Vital Monitor")
    
    with gr.Tabs():
        with gr.TabItem("Live Measurement"):
            with gr.Row():
                in_video = gr.Image(sources="webcam", streaming=True, label="Live Feed")
                with gr.Column():
                    out_video = gr.Image(label="Processed Signal")
                    status_text = gr.Markdown("Position your forehead in the blue box.")
                    stop_btn = gr.Button("Finish & See Report", variant="primary")
            
            in_video.stream(
                fn=process_vitals,
                inputs=[in_video, state],
                outputs=[out_video, state, status_text]
            )
            
        with gr.TabItem("Final Report"):
            report_area = gr.Markdown("Measurement summary will appear here.")
            stop_btn.click(fn=show_results, inputs=[state], outputs=[report_area])

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
