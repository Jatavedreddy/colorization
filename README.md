# Image Colorization using PyTorch

This repository contains PyTorch implementations of image colorization models.

## Getting Started

### Prerequisites
- Python 3
- PyTorch
- Numpy
- Scikit-image
- Pillow
- opencv-python
- moviepy
- Gradio

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/richzhang/colorization.git
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the `app.py` script:
```
python app.py
```
This will launch an interactive **Gradio** interface in your browser with three tabs:

1. **Single Image** – Upload an image (optionally a ground-truth colour image), choose a colourization strength, and interactively compare results using draggable sliders. PSNR/SSIM metrics are displayed if ground-truth is provided.
2. **Batch** – Drop in a set of images and download a ZIP archive containing colourized versions (both ECCV16 and SIGGRAPH17) alongside gallery previews.
3. **Video** – Upload a short MP4 clip; each frame is colourized (ECCV16) and re-assembled into a downloadable video.

Behind the scenes the app:

* keeps the original resolution of your image (no more 256 px blur!),
* caches repeated requests for instant responses,
* computes quantitative metrics (PSNR, SSIM) when possible, and
* supports GPU acceleration automatically if CUDA is available.
