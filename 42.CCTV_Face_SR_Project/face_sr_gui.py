"""
CCTV Face Super-Resolution GUI
-------------------------------
Simple GUI to compare Bicubic vs FSRCNN-small (x2) on face images.
Upload a face photo → Click process → View before & after.
"""

import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
import os

# ---------- Load FSRCNN Model ----------
MODEL_PATH = "FSRCNN_x2.pb"

# If FSRCNN model is missing, show warning
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Model Missing", "FSRCNN_x2.pb not found in project folder!\nPlease add it and restart.")
    exit()

# Load model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_PATH)
sr.setModel("fsrcnn", 2)
print("[INFO] FSRCNN model loaded successfully.")

# ---------- GUI Class ----------
class FaceSRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Face Super-Resolution (Bicubic vs FSRCNN)")
        self.root.geometry("1100x650")
        self.root.config(bg="#d8f3dc")  # light green theme

        # Title label
        self.label = Label(root, text="CCTV Face Super-Resolution", font=("Arial", 20, "bold"), bg="#d8f3dc", fg="#1b4332")
        self.label.pack(pady=15)

        # Upload Button
        self.btn_upload = Button(root, text="📂 Upload Face Image", command=self.load_image,
                                 font=("Arial", 14, "bold"), bg="#52b788", fg="white", padx=20, pady=10)
        self.btn_upload.pack(pady=10)

        # Process Button
        self.btn_process = Button(root, text="🚀 Enhance Resolution", command=self.run_sr,
                                  font=("Arial", 14, "bold"), bg="#40916c", fg="white", padx=20, pady=10)
        self.btn_process.pack(pady=10)

        # Image Display Area
        self.display_label = Label(root, bg="#d8f3dc")
        self.display_label.pack(pady=15)

        # Info label
        self.info_label = Label(root, text="", font=("Arial", 12), bg="#d8f3dc", fg="#081c15")
        self.info_label.pack()

        self.image_path = None

    # Step 1: Load an image
    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img_tk = ImageTk.PhotoImage(Image.fromarray(img))
            self.display_label.configure(image=img_tk)
            self.display_label.image = img_tk
            self.info_label.config(text=f"Loaded: {os.path.basename(self.image_path)}")

    # Step 2: Run Super-Resolution
    def run_sr(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload a face image first!")
            return

        # Read & preprocess
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Create a low-res version (simulate CCTV)
        lr = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

        # Bicubic Upscale
        bicubic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        # FSRCNN Upscale
        fsrcnn = sr.upsample(lr)

        # Ensure both outputs are same size
        fsrcnn = cv2.resize(fsrcnn, (w, h))

        # Combine both images side-by-side
        combined = np.concatenate([bicubic, fsrcnn], axis=1)
        combined_resized = cv2.resize(combined, (900, 450))

        # Convert to Tkinter Image
        combined_tk = ImageTk.PhotoImage(Image.fromarray(combined_resized))
        self.display_label.configure(image=combined_tk)
        self.display_label.image = combined_tk
        self.info_label.config(text="Bicubic (Left)  |  FSRCNN (Right)")

        # Save output automatically
        out_path = "output_gui_result.png"
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved result to {out_path}")

# ---------- Main Run ----------
if __name__ == "__main__":
    root = Tk()
    app = FaceSRGUI(root)
    root.mainloop()
