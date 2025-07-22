import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from colorizers import eccv16, siggraph17

# ------------------------------
# Model Loading (once)
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colorizer_eccv16 = eccv16(pretrained=True).to(device).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).to(device).eval()

# ------------------------------
# Simple in-memory cache
# ------------------------------
# key: sha256(image_bytes + strength) -> (eccv_img, sig_img)
_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _sha_key(arr: np.ndarray, strength: float) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update(str(strength).encode())
    return h.hexdigest()


# ------------------------------
# Core colorization utilities
# ------------------------------

def _predict_ab(model, l_channel: np.ndarray) -> np.ndarray:
    """Runs the model on a 256×256 L-channel and returns ab [H,W,2] in range-128…128."""
    tens = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_ab = model(tens)
    ab = (
        out_ab.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )  # (256,256,2), still ‑128…128
    return ab


def colorize_highres(image_np: np.ndarray, strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Colorize image at its native resolution with both models and blend with the original.

    Returns two arrays in RGB float [0,1] matching the original resolution.
    """
    # Ensure float in [0,1]
    if image_np.dtype != np.float64 and image_np.dtype != np.float32:
        img = image_np.astype(np.float64) / 255.0
    else:
        img = np.clip(image_np, 0, 1)

    h, w = img.shape[:2]

    # Extract original L channel
    img_lab_orig = rgb2lab(img)
    img_l_orig = img_lab_orig[:, :, 0]

    # Down-scale to 256×256 for model input
    img_small = resize(img, (256, 256), preserve_range=True)
    img_lab_small = rgb2lab(img_small)
    img_l_small = img_lab_small[:, :, 0]

    # Predict ab for both models
    ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
    ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)

    # Upsample ab maps to original resolution
    ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
    ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)

    def _lab_to_rgb(l_orig: np.ndarray, ab_up: np.ndarray) -> np.ndarray:
        lab = np.concatenate((l_orig[:, :, np.newaxis], ab_up), axis=2)
        rgb = lab2rgb(lab)
        return rgb

    out_eccv_rgb = _lab_to_rgb(img_l_orig, ab_eccv_up)
    out_sig_rgb = _lab_to_rgb(img_l_orig, ab_sig_up)

    # Blend with original for fine detail
    strength = float(strength)
    out_eccv_rgb = (1 - strength) * img + strength * out_eccv_rgb
    out_sig_rgb = (1 - strength) * img + strength * out_sig_rgb

    return out_eccv_rgb, out_sig_rgb


def get_colorized(image_np: np.ndarray, strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """Colorize image with caching."""
    key = _sha_key(image_np, strength)
    if key in _cache:
        return _cache[key]
    eccv_img, sig_img = colorize_highres(image_np, strength)
    _cache[key] = (eccv_img, sig_img)
    return eccv_img, sig_img


# ------------------------------
# Metric utilities
# ------------------------------

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Returns (PSNR, SSIM) for images in RGB float [0,1]."""
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
    return psnr_val, ssim_val


# ------------------------------
# Gradio Handlers
# ------------------------------

def handler_single(input_img: np.ndarray, strength: float, gt_img: np.ndarray | None):
    if input_img is None:
        return None, None, "<span style='color:red'>Please upload an image.</span>"

    eccv_img, sig_img = get_colorized(input_img, strength)

    slider_eccv = (input_img, (eccv_img * 255).astype(np.uint8))
    slider_sig = (input_img, (sig_img * 255).astype(np.uint8))

    metrics_html = "<b>No ground-truth supplied &ndash; metrics not computed.</b>"
    if gt_img is not None:
        # Resize GT if necessary
        if gt_img.shape[:2] != input_img.shape[:2]:
            gt_resized = resize(gt_img, input_img.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            gt_resized = gt_img
        gt_float = gt_resized.astype(np.float64) / 255.0
        eccv_float = eccv_img
        sig_float = sig_img
        psnr_eccv, ssim_eccv = compute_metrics(gt_float, eccv_float)
        psnr_sig, ssim_sig = compute_metrics(gt_float, sig_float)
        metrics_html = f"""<table>
        <tr><th></th><th>PSNR</th><th>SSIM</th></tr>
        <tr><td>ECCV16</td><td>{psnr_eccv:.2f}</td><td>{ssim_eccv:.3f}</td></tr>
        <tr><td>SIGGRAPH17</td><td>{psnr_sig:.2f}</td><td>{ssim_sig:.3f}</td></tr>
        </table>"""

    return slider_eccv, slider_sig, metrics_html


# ------------------------------
# Batch processing
# ------------------------------

def handler_batch(files: List[str] | None, strength: float):
    if not files:
        return [], [], None

    eccv_gallery, sig_gallery = [], []

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tmp_zip, "w") as zf:
        for file_path in files:
            file_path = Path(file_path)
            try:
                img = Image.open(file_path).convert("RGB")
            except Exception:
                continue
            img_np = np.array(img)
            eccv_img, sig_img = get_colorized(img_np, strength)

            eccv_uint8 = (eccv_img * 255).astype(np.uint8)
            sig_uint8 = (sig_img * 255).astype(np.uint8)

            eccv_gallery.append(eccv_uint8)
            sig_gallery.append(sig_uint8)

            base = file_path.stem
            eccv_name = f"{base}_eccv16.png"
            sig_name = f"{base}_siggraph17.png"

            # Save into ZIP
            with io.BytesIO() as buff:
                Image.fromarray(eccv_uint8).save(buff, format="PNG")
                zf.writestr(eccv_name, buff.getvalue())
            with io.BytesIO() as buff:
                Image.fromarray(sig_uint8).save(buff, format="PNG")
                zf.writestr(sig_name, buff.getvalue())

    tmp_zip.flush()
    return eccv_gallery, sig_gallery, tmp_zip.name


# ------------------------------
# Video processing
# ------------------------------

def handler_video(video_file: str | None, strength: float):
    if video_file is None:
        return None

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temporary output file
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        eccv_img, _ = get_colorized(frame_rgb, strength)  # use ECCV16 for speed
        out_frame_bgr = cv2.cvtColor((eccv_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(out_frame_bgr)

    cap.release()
    writer.release()
    return out_path


# ------------------------------
# Build Gradio UI
# ------------------------------

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """# 🖼️ Image & Video Colorization Demo

This upgraded demo offers **high-resolution colorization**, **batch processing**, **ground-truth evaluation**, **video support**, interactive **image sliders**, and smart **caching**.
"""
        )

        with gr.Tabs():
            # ----- Single Image -----
            with gr.TabItem("Single Image"):
                with gr.Row():
                    inp_img = gr.Image(type="numpy", label="Input Image")
                    gt_img = gr.Image(type="numpy", label="Ground-truth Image (optional)")
                strength_slider = gr.Slider(0, 1, value=1.0, step=0.1, label="Colorization Strength")
                run_btn = gr.Button("Colorize")

                slider_eccv = gr.ImageSlider(label="Original ↔ ECCV16")
                slider_sig = gr.ImageSlider(label="Original ↔ SIGGRAPH17")
                metrics_html = gr.HTML()

                run_btn.click(
                    handler_single,
                    inputs=[inp_img, strength_slider, gt_img],
                    outputs=[slider_eccv, slider_sig, metrics_html],
                )

            # ----- Batch -----
            with gr.TabItem("Batch"):
                files_input = gr.Files(label="Upload multiple images")
                strength_slider2 = gr.Slider(0, 1, value=1.0, step=0.1, label="Colorization Strength")
                run_batch = gr.Button("Colorize Batch")

                gallery_eccv = gr.Gallery(label="ECCV16 Outputs", columns=4, height="auto")
                gallery_sig = gr.Gallery(label="SIGGRAPH17 Outputs", columns=4, height="auto")
                zip_out = gr.File(label="Download ZIP")

                run_batch.click(
                    handler_batch,
                    inputs=[files_input, strength_slider2],
                    outputs=[gallery_eccv, gallery_sig, zip_out],
                )

            # ----- Video -----
            with gr.TabItem("Video"):
                vid_input = gr.File(label="Upload MP4 video")
                strength_slider3 = gr.Slider(0, 1, value=1.0, step=0.1, label="Colorization Strength")
                run_vid = gr.Button("Colorize Video (ECCV16)")
                vid_out = gr.Video()

                run_vid.click(
                    handler_video,
                    inputs=[vid_input, strength_slider3],
                    outputs=vid_out,
                )

    return demo


if __name__ == "__main__":
    iface = build_interface()
    iface.launch() 