import gradio as gr
import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from colorizers import eccv16, siggraph17

# Load models
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

def colorize(image, colorization_strength):
    # Preprocess the image
    img_resized = resize(image, (256, 256)).astype(np.float64)
    
    # to Lab
    img_lab = rgb2lab(img_resized)
    img_l = img_lab[:,:,0]
    
    # to tensor
    img_l_tensor = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float()

    # Colorize with ECCV16
    with torch.no_grad():
        out_ab_eccv16 = colorizer_eccv16(img_l_tensor)
    out_lab_eccv16 = torch.cat((img_l_tensor, out_ab_eccv16), dim=1).squeeze().permute(1,2,0).numpy()
    out_rgb_eccv16 = lab2rgb(out_lab_eccv16)

    # Colorize with SIGGRAPH17
    with torch.no_grad():
        out_ab_siggraph17 = colorizer_siggraph17(img_l_tensor)
    out_lab_siggraph17 = torch.cat((img_l_tensor, out_ab_siggraph17), dim=1).squeeze().permute(1,2,0).numpy()
    out_rgb_siggraph17 = lab2rgb(out_lab_siggraph17)

    # Blend with original
    strength = float(colorization_strength)
    out_rgb_eccv16 = (1 - strength) * img_resized + strength * out_rgb_eccv16
    out_rgb_siggraph17 = (1 - strength) * img_resized + strength * out_rgb_siggraph17
    
    return out_rgb_eccv16, out_rgb_siggraph17

# Create the Gradio interface
iface = gr.Interface(
    fn=colorize,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Slider(minimum=0, maximum=1, step=0.1, value=1.0, label="Colorization Strength")
    ],
    outputs=[
        gr.Image(type="numpy", label="ECCV16 Output"),
        gr.Image(type="numpy", label="SIGGRAPH17 Output")
    ],
    title="Image Colorization",
    description="Upload an image to colorize it using two different models. Adjust the colorization strength to blend the output with the original image.",
    live=True
)

iface.launch() 