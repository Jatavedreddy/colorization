import argparse
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from colorizers import eccv16, siggraph17

def colorize_image(image_path):
    # Load models
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()

    # Preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    img_resized = resize(img, (256, 256))
    
    # to Lab
    img_lab = rgb2lab(img_resized)
    img_l = img_lab[:,:,0]
    
    # to tensor
    img_l_tensor = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float()

    # Process and colorize the image
    # ECCV16
    with torch.no_grad():
        out_ab_eccv16 = colorizer_eccv16(img_l_tensor)
    
    # combine L and ab
    out_lab_eccv16 = torch.cat((img_l_tensor, out_ab_eccv16), dim=1).squeeze().permute(1,2,0).numpy()
    out_rgb_eccv16 = lab2rgb(out_lab_eccv16)
    plt.imsave('imgs_out/colorized_eccv16.png', out_rgb_eccv16)

    # SIGGRAPH17
    with torch.no_grad():
        out_ab_siggraph17 = colorizer_siggraph17(img_l_tensor)

    out_lab_siggraph17 = torch.cat((img_l_tensor, out_ab_siggraph17), dim=1).squeeze().permute(1,2,0).numpy()
    out_rgb_siggraph17 = lab2rgb(out_lab_siggraph17)
    plt.imsave('imgs_out/colorized_siggraph17.png', out_rgb_siggraph17)


    print(f"Colorized images saved in 'imgs_out' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize a single image.')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()
    
    colorize_image(args.image_path) 