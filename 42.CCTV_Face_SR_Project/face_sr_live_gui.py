"""
CCTV Face Super-Resolution - Full GUI (Live + Upload)
-----------------------------------------------------
Features:
 - Start webcam and capture a frame
 - Upload existing face image
 - Detect face, crop, upscale (Bicubic + FSRCNN)
 - Show side-by-side results with sharpening
"""

import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
import os

# ---------- Paths ----------
MODEL_PATH = "FSRCNN_x2.pb"
FACE_PATH = os.path.join("models", "haarcascade_frontalface_default.xml")

# ---------- Model Checks ----------
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Missing Model", "Please place FSRCNN_x2.pb in the project folder.")
    raise SystemExit

if not os.path.exists(FACE_PATH):
    FACE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(FACE_PATH)

# ---------- Load FSRCNN ----------
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("fsrcnn", 2)
print("[INFO] FSRCNN model loaded.")

# ---------- Helper Functions ----------
def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharp = cv2.addWeighted(img, 1.3, blur, -0.3, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def detect_face(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    (x, y, w, h) = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
    pad = int(0.15 * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame_rgb.shape[1], x + w + pad), min(frame_rgb.shape[0], y + h + pad)
    return frame_rgb[y1:y2, x1:x2]

# ---------- GUI ----------
class FaceSRDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Face Super-Resolution (Live + Upload)")
        self.root.geometry("1250x750")
        self.root.config(bg="#d8f3dc")

        Label(root, text="CCTV Face Super-Resolution (FSRCNN + Bicubic)",
              font=("Arial", 20, "bold"), bg="#d8f3dc", fg="#1b4332").pack(pady=15)

        # Buttons
        Button(root, text="▶ Start Camera", command=self.start_camera,
               font=("Arial", 14, "bold"), bg="#52b788", fg="white", padx=15, pady=8).pack(pady=5)
        Button(root, text="📸 Capture & Enhance", command=self.capture_frame,
               font=("Arial", 14, "bold"), bg="#40916c", fg="white", padx=15, pady=8).pack(pady=5)
        Button(root, text="📂 Upload Image & Enhance", command=self.upload_image,
               font=("Arial", 14, "bold"), bg="#74c69d", fg="black", padx=15, pady=8).pack(pady=5)
        Button(root, text="❌ Exit", command=self.exit_app,
               font=("Arial", 14, "bold"), bg="#d00000", fg="white", padx=15, pady=8).pack(pady=5)

        # Video + Output labels
        self.video_label = Label(root, bg="#d8f3dc")
        self.video_label.pack(pady=10)
        self.result_label = Label(root, bg="#d8f3dc")
        self.result_label.pack(pady=10)
        self.status_label = Label(root, text="", bg="#d8f3dc", fg="#081c15", font=("Arial", 11))
        self.status_label.pack()

        self.cap = None
        self.frame = None
        self.running = False

    # ---------- Camera ----------
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Webcam not found!")
            return
        self.running = True
        self.show_video()

    def show_video(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview = cv2.resize(frame_rgb, (480, 360))
                img_tk = ImageTk.PhotoImage(Image.fromarray(preview))
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk
                self.frame = frame_rgb
            self.root.after(15, self.show_video)

    # ---------- Capture from Live ----------
    def capture_frame(self):
        if self.frame is None:
            messagebox.showwarning("No Frame", "Start camera first.")
            return
        self.process_image(self.frame, source="Live Frame")

    # ---------- Upload from File ----------
    def upload_image(self):
        path = filedialog.askopenfilename(title="Select Face Image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.process_image(img, source=os.path.basename(path))

    # ---------- Common Processing ----------
    def process_image(self, img_rgb, source="Image"):
        face = detect_face(img_rgb)
        if face is None:
            self.status_label.config(text="⚠ No face detected! Try clearer or closer image.")
            return

        h, w = face.shape[:2]
        if min(h, w) < 160:
            scale = 200 / min(h, w)
            face = cv2.resize(face, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

        lr = cv2.resize(face, (face.shape[1]//2, face.shape[0]//2), interpolation=cv2.INTER_CUBIC)
        bicubic = cv2.resize(lr, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_CUBIC)
        fsrcnn = sr.upsample(lr)
        fsrcnn = cv2.resize(fsrcnn, (face.shape[1], face.shape[0]))
        fsrcnn_sharp = sharpen(fsrcnn)

        combined = np.concatenate([face, bicubic, fsrcnn_sharp], axis=1)
        disp = cv2.resize(combined, (1000, 320))
        img_tk = ImageTk.PhotoImage(Image.fromarray(disp))
        self.result_label.configure(image=img_tk)
        self.result_label.image = img_tk

        out_path = f"enhanced_{source.replace(' ', '_')}.png"
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        self.status_label.config(text=f"✅ Enhanced & saved: {out_path}\nLeft: Original | Mid: Bicubic | Right: FSRCNN (Sharpened)")

    # ---------- Exit ----------
    def exit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

# ---------- Run ----------
if __name__ == "__main__":
    root = Tk()
    app = FaceSRDemo(root)
    root.mainloop()
