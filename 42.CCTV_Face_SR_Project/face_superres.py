"""
CCTV Face Super-Resolution PBL Project
--------------------------------------
This script compares Bicubic vs FSRCNN-small (x2) for face image upscaling.
Includes auto-resizing small images to avoid SSIM errors.

Usage:
  python face_superres.py --input my_faces --output outputs

Requirements:
  pip install numpy opencv-python opencv-contrib-python matplotlib scikit-image face-recognition
"""

import os
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import csv
import urllib.request
import shutil

# Optional imports for face similarity
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    FACE_REC_AVAILABLE = False

# ---------------------------
# Configurations
# ---------------------------
FSRCNN_MODEL_URL = "https://github.com/Saafke/FSRCNN-tensorflow/raw/master/models/FSRCNN_x2.pb"
FSRCNN_MODEL_LOCAL = "FSRCNN_x2.pb"

# ---------------------------
# Utility Functions
# ---------------------------
def download_model(url=FSRCNN_MODEL_URL, dest=FSRCNN_MODEL_LOCAL):
    if os.path.exists(dest):
        print(f"[INFO] FSRCNN model already exists: {dest}")
        return dest
    print(f"[INFO] Downloading FSRCNN model...")
    try:
        with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("[INFO] Download complete.")
        return dest
    except Exception as e:
        print("[ERROR] Could not download FSRCNN model:", e)
        return None

def ensure_output_dirs(out_root):
    paths = {
        "lr": os.path.join(out_root, "LR"),
        "bicubic": os.path.join(out_root, "bicubic"),
        "fsrcnn": os.path.join(out_root, "fsrcnn"),
        "sidebyside": os.path.join(out_root, "side_by_side"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_rgb(path, img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def create_lr_from_hr(hr_rgb):
    h, w = hr_rgb.shape[:2]
    lr = cv2.resize(hr_rgb, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
    return lr

def bicubic_upscale(lr_rgb, target_size):
    up = cv2.resize(lr_rgb, target_size, interpolation=cv2.INTER_CUBIC)
    return up

def load_fsrcnn_model(model_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 2)
    return sr

def fsrcnn_upscale(sr_model, lr_rgb):
    lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = sr_model.upsample(lr_bgr)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

def compute_metrics(hr_rgb, pred_rgb):
    hr = np.asarray(hr_rgb).astype(np.float32)
    pr = np.asarray(pred_rgb).astype(np.float32)
    psnr_val = sk_psnr(hr, pr, data_range=255)
    ssim_val = sk_ssim(hr, pr, channel_axis=-1, data_range=255)
    return psnr_val, ssim_val

def compute_face_similarity(hr_rgb, pred_rgb):
    if not FACE_REC_AVAILABLE:
        return None
    try:
        enc_hr = face_recognition.face_encodings(hr_rgb)
        enc_pr = face_recognition.face_encodings(pred_rgb)
        if len(enc_hr) == 0 or len(enc_pr) == 0:
            return None
        e1 = enc_hr[0]
        e2 = enc_pr[0]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1)*np.linalg.norm(e2) + 1e-10)
        return float(cos_sim)
    except Exception as e:
        print("[WARN] face_recognition error:", e)
        return None

def make_side_by_side(lr_rgb, bic_rgb, fsrcnn_rgb, hr_rgb):
    hr_h, hr_w = hr_rgb.shape[:2]
    lr_vis = cv2.resize(lr_rgb, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([lr_vis, bic_rgb, fsrcnn_rgb, hr_rgb], axis=1)

# ---------------------------
# Main Processing
# ---------------------------
def process_image_file(path, out_dirs, sr_model=None, do_face_id=False):
    basename = os.path.splitext(os.path.basename(path))[0]
    hr_rgb = read_image(path)

    # 🟢 Resize too-small images to avoid SSIM error
    h, w = hr_rgb.shape[:2]
    if h < 64 or w < 64:
        scale = 128 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        hr_rgb = cv2.resize(hr_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"[INFO] Resized small image to {new_w}x{new_h} for processing.")

    h, w = hr_rgb.shape[:2]
    lr_rgb = create_lr_from_hr(hr_rgb)
    bic_rgb = bicubic_upscale(lr_rgb, (w, h))

    if sr_model is not None:
        try:
            fsrcnn_rgb = fsrcnn_upscale(sr_model, lr_rgb)
        except Exception as e:
            print("[ERROR] FSRCNN upsample failed:", e)
            fsrcnn_rgb = bic_rgb.copy()
    else:
        fsrcnn_rgb = bic_rgb.copy()

    # Compute metrics
    psnr_b, ssim_b = compute_metrics(hr_rgb, bic_rgb)
    psnr_f, ssim_f = compute_metrics(hr_rgb, fsrcnn_rgb)
    face_sim_b = compute_face_similarity(hr_rgb, bic_rgb) if do_face_id else None
    face_sim_f = compute_face_similarity(hr_rgb, fsrcnn_rgb) if do_face_id else None

    # Save outputs
    save_image_rgb(os.path.join(out_dirs["lr"], f"{basename}_LR.png"), lr_rgb)
    save_image_rgb(os.path.join(out_dirs["bicubic"], f"{basename}_bic.png"), bic_rgb)
    save_image_rgb(os.path.join(out_dirs["fsrcnn"], f"{basename}_fsrcnn.png"), fsrcnn_rgb)
    side = make_side_by_side(lr_rgb, bic_rgb, fsrcnn_rgb, hr_rgb)
    save_image_rgb(os.path.join(out_dirs["sidebyside"], f"{basename}_side_by_side.png"), side)

    return {
        "image": basename,
        "psnr_bicubic": psnr_b,
        "ssim_bicubic": ssim_b,
        "psnr_fsrcnn": psnr_f,
        "ssim_fsrcnn": ssim_f,
        "face_sim_bicubic": face_sim_b,
        "face_sim_fsrcnn": face_sim_f
    }

# ---------------------------
# CLI / Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CCTV Face Super-Resolution (Bicubic vs FSRCNN-small)")
    parser.add_argument("--input", "-i", required=True, help="Input image or folder path")
    parser.add_argument("--output", "-o", default="outputs", help="Output folder (default: outputs)")
    parser.add_argument("--no-face-id", action="store_true", help="Disable face ID similarity check")
    parser.add_argument("--download-model", action="store_true", help="Force download FSRCNN model")
    return parser.parse_args()

def gather_input_images(inp):
    if os.path.isdir(inp):
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        files = [os.path.join(inp, f) for f in os.listdir(inp) if os.path.splitext(f)[1].lower() in exts]
        files.sort()
        return files
    elif os.path.isfile(inp):
        return [inp]
    else:
        raise ValueError("Invalid input path: " + inp)

def main():
    args = parse_args()
    out_dirs = ensure_output_dirs(args.output)

    model_path = FSRCNN_MODEL_LOCAL if os.path.exists(FSRCNN_MODEL_LOCAL) else download_model()
    sr_model = None
    if model_path:
        try:
            sr_model = load_fsrcnn_model(model_path)
            print("[INFO] FSRCNN model loaded.")
        except Exception as e:
            print("[WARN] Could not load FSRCNN:", e)

    do_face_id = (not args.no_face_id) and FACE_REC_AVAILABLE
    if args.no_face_id:
        print("[INFO] Face ID similarity disabled.")
    elif not FACE_REC_AVAILABLE:
        print("[INFO] face_recognition not available, skipping ID similarity.")

    files = gather_input_images(args.input)
    if len(files) == 0:
        print("[ERROR] No images found in input.")
        return

    results = []
    print(f"[INFO] Processing {len(files)} images... (outputs -> {args.output})")

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f}")
        try:
            result = process_image_file(f, out_dirs, sr_model=sr_model, do_face_id=do_face_id)
            results.append(result)
        except Exception as e:
            print("[ERROR] Failed processing", f, ":", e)

    csv_path = os.path.join(args.output, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[INFO] Done. Results saved to: {csv_path}")
    print(f"[INFO] Side-by-side images in: {out_dirs['sidebyside']}")

if __name__ == "__main__":
    main()
