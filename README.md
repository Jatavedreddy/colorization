# Image Colorization using PyTorch

This repository contains PyTorch implementations of image colorization models.

## Getting Started

### Prerequisites
- Python 3
- PyTorch
- Scikit-image
- Numpy
- Matplotlib

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

To colorize an image, run the `demo.py` script with the path to your image:
```
python demo.py -i <path_to_your_image>
```
For example:
```
python demo.py -i imgs/ansel_adams.jpg
```
The colorized images will be saved in the `imgs_out` directory.
