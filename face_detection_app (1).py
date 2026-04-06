import gradio as gr
import cv2
import numpy as np
from PIL import Image
import threading

# Load cascades once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ── Frame-skip state (per-session via a simple shared dict) ──────────────────
_state = {"frame_count": 0, "last_faces": [], "lock": threading.Lock()}

DETECT_EVERY_N = 3        # run detection only every N frames
DETECT_WIDTH   = 320      # downscale to this width for detection only


def draw_faces(img_bgr, faces, gray, detect_eyes, draw_style):
    colors = [(0,255,0),(255,100,0),(0,100,255),(255,0,255),(0,255,255)]
    eye_count = 0
    for i, (x, y, w, h) in enumerate(faces):
        color = colors[i % len(colors)]
        if draw_style == "Rectangle":
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        elif draw_style == "Rounded":
            r = max(1, min(w, h) // 8)
            cv2.rectangle(img_bgr, (x+r, y),   (x+w-r, y+h), color, 2)
            cv2.rectangle(img_bgr, (x,   y+r),  (x+w, y+h-r), color, 2)
            cv2.ellipse(img_bgr, (x+r,   y+r),   (r,r), 180, 0, 90, color, 2)
            cv2.ellipse(img_bgr, (x+w-r, y+r),   (r,r), 270, 0, 90, color, 2)
            cv2.ellipse(img_bgr, (x+r,   y+h-r), (r,r),  90, 0, 90, color, 2)
            cv2.ellipse(img_bgr, (x+w-r, y+h-r), (r,r),   0, 0, 90, color, 2)
        elif draw_style == "Ellipse":
            cv2.ellipse(img_bgr, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, color, 2)
        cv2.putText(img_bgr, f"Face {i+1}", (x, max(0,y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if detect_eyes and gray is not None:
            roi_gray = gray[y:y+h, x:x+w]
            roi_bgr  = img_bgr[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15,15))
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(roi_bgr, (ex+ew//2, ey+eh//2), ew//2, (255,200,0), 2)
                eye_count += 1
    return eye_count


def detect_faces(image, scale_factor=1.1, min_neighbors=5, min_size=30,
                 detect_eyes=False, draw_style="Rectangle"):
    if image is None:
        return None, "No image provided."

    img_np  = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    with _state["lock"]:
        _state["frame_count"] += 1
        run_detection = (_state["frame_count"] % DETECT_EVERY_N == 0)

    if run_detection:
        # Downscale for fast detection
        scale  = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
        small  = cv2.resize(img_bgr, (int(W*scale), int(H*scale)),
                            interpolation=cv2.INTER_LINEAR)
        gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.equalizeHist(gray_s)

        faces_small = face_cascade.detectMultiScale(
            gray_s,
            scaleFactor=float(scale_factor),
            minNeighbors=int(min_neighbors),
            minSize=(int(min_size * scale), int(min_size * scale)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Scale boxes back to original resolution
        if len(faces_small) > 0:
            faces = (faces_small / scale).astype(int)
            faces[:, 2] = np.minimum(faces[:, 0] + faces[:, 2], W) - faces[:, 0]
            faces[:, 3] = np.minimum(faces[:, 1] + faces[:, 3], H) - faces[:, 1]
        else:
            faces = []

        with _state["lock"]:
            _state["last_faces"] = faces
    else:
        with _state["lock"]:
            faces = _state["last_faces"]

    face_count = len(faces)

    # Draw on full-res frame
    gray_full = None
    if detect_eyes and face_count > 0:
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    eye_count = draw_faces(img_bgr, faces, gray_full, detect_eyes, draw_style)

    # Overlay FPS-friendly status badge
    label = f"Faces: {face_count}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_bgr, (8, 8), (18+tw, 18+th), (0,0,0), -1)
    cv2.putText(img_bgr, label, (13, 13+th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,128), 2, cv2.LINE_AA)

    result = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    if face_count == 0:
        summary = "😶 No faces detected."
    else:
        summary = f"✅ **{face_count}** face{'s' if face_count > 1 else ''} detected"
        if detect_eyes:
            summary += f" · **{eye_count}** eye{'s' if eye_count != 1 else ''}"
        summary += "."

    return result, summary


def settings_block():
    with gr.Accordion("⚙️ Detection Settings", open=False):
        sf = gr.Slider(1.05, 1.5, value=1.1, step=0.01, label="Scale Factor",
                       info="Lower = more sensitive but slower.")
        mn = gr.Slider(1, 15, value=5, step=1, label="Min Neighbors",
                       info="Higher = fewer but more accurate detections.")
        ms = gr.Slider(10, 150, value=30, step=5, label="Min Face Size (px)")
        de = gr.Checkbox(label="Also detect eyes 👀", value=False)
        ds = gr.Radio(["Rectangle","Rounded","Ellipse"], value="Rectangle",
                      label="Bounding-box style")
    return sf, mn, ms, de, ds


with gr.Blocks(title="Real-Time Face Detection") as demo:

    gr.HTML("""
        <div style="text-align:center;padding:16px 0 4px">
            <h1 style="font-size:2rem;font-weight:700;color:#065f46">🎭 Real-Time Face Detection</h1>
            <p style="color:#6b7280;margin-top:4px">Webcam live detection or upload a photo.</p>
        </div>
    """)

    with gr.Tabs():

        with gr.TabItem("📷 Webcam (Live)"):
            with gr.Row():
                with gr.Column():
                    webcam_in = gr.Image(
                        sources=["webcam"],
                        type="pil",
                        label="Webcam",
                        streaming=True,
                        mirror_webcam=True,
                    )
                    sf1, mn1, ms1, de1, ds1 = settings_block()
                with gr.Column():
                    webcam_out  = gr.Image(label="Detection Result", type="pil")
                    webcam_text = gr.Markdown()

            webcam_in.stream(
                fn=detect_faces,
                inputs=[webcam_in, sf1, mn1, ms1, de1, ds1],
                outputs=[webcam_out, webcam_text],
                time_limit=120,
                stream_every=0.066,   # ~15 fps cap → smooth without overload
            )

        with gr.TabItem("🖼️ Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_in  = gr.Image(sources=["upload"], type="pil",
                                          label="Upload an image")
                    sf2, mn2, ms2, de2, ds2 = settings_block()
                    detect_btn = gr.Button("🔍 Detect Faces", variant="primary", size="lg")
                with gr.Column():
                    upload_out  = gr.Image(label="Detection Result", type="pil")
                    upload_text = gr.Markdown()

            detect_btn.click(
                fn=detect_faces,
                inputs=[upload_in, sf2, mn2, ms2, de2, ds2],
                outputs=[upload_out, upload_text],
            )
            upload_in.change(
                fn=detect_faces,
                inputs=[upload_in, sf2, mn2, ms2, de2, ds2],
                outputs=[upload_out, upload_text],
            )

    gr.Markdown("""
### 💡 Tips
- **Still laggy?** Lower *Min Face Size* or raise *Scale Factor* to speed up detection.  
- **Missing faces?** Lower *Scale Factor* (try 1.05) or reduce *Min Neighbors*.  
- **Too many false positives?** Raise *Min Neighbors* to 7–10.
""")

if __name__ == "__main__":
    demo.launch()
