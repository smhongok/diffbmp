import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import data, color, io
from skimage.transform import resize
from skimage.util import img_as_float
import json
import argparse
import os

# Argument parser setup
parser = argparse.ArgumentParser(description="Process images with Structure-Aware Graphics Synthesis")
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()
config_path = args.config

# Load configuration
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Load the image from the path specified in the config
img_path = config["preprocessing"]["img_path"]
image = io.imread(img_path)  # Load the image first

# Handle RGBA images by converting to RGB first
if image.shape[-1] == 4:  # If image has 4 channels (RGBA)
    # Convert RGBA to RGB by removing the alpha channel
    image = image[:, :, :3]

# Resize the original image
image_resized = resize(image, (256, 256))
image_resized = img_as_float(image_resized)

# Convert to grayscale for grayscale processing
image_gray = color.rgb2gray(image_resized)

# Perform 2D discrete wavelet transform on grayscale
wavelet = 'db1'
coeffs2_gray = pywt.wavedec2(image_gray, wavelet=wavelet, level=3)
coeff_arr_gray, coeff_slices_gray = pywt.coeffs_to_array(coeffs2_gray)

# Sort coefficients by magnitude and retain top-k
def reconstruct_from_top_k(coeff_arr, coeff_slices, k):
    flat = coeff_arr.flatten()
    indices = np.argsort(np.abs(flat))[::-1]
    mask = np.zeros_like(flat)
    mask[indices[:k]] = 1
    masked_arr = flat * mask
    masked_coeffs = masked_arr.reshape(coeff_arr.shape)
    masked_coeffs_struct = pywt.array_to_coeffs(masked_coeffs, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(masked_coeffs_struct, wavelet=wavelet)

# Reconstruct grayscale images with different numbers of coefficients
ks = [1000, 2000, 5000, 10000, coeff_arr_gray.size]
recon_images_gray = [reconstruct_from_top_k(coeff_arr_gray, coeff_slices_gray, k) for k in ks]

# Process color image - apply wavelet transform to each channel separately
def process_color_image(image, k):
    # Process each color channel separately
    channels = []
    for i in range(3):  # RGB channels
        channel = image[:, :, i]
        coeffs2 = pywt.wavedec2(channel, wavelet=wavelet, level=3)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs2)
        
        # Reconstruct with top k coefficients
        flat = coeff_arr.flatten()
        indices = np.argsort(np.abs(flat))[::-1]
        mask = np.zeros_like(flat)
        mask[indices[:k]] = 1
        masked_arr = flat * mask
        masked_coeffs = masked_arr.reshape(coeff_arr.shape)
        masked_coeffs_struct = pywt.array_to_coeffs(masked_coeffs, coeff_slices, output_format='wavedec2')
        reconstructed = pywt.waverec2(masked_coeffs_struct, wavelet=wavelet)
        channels.append(reconstructed)
    
    # Combine channels back into a color image
    return np.stack(channels, axis=2)

# Reconstruct color images with different numbers of coefficients
recon_images_color = [process_color_image(image_resized, k) for k in ks]

# Normalize color images to ensure they're in the valid range (0 to 1)
def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if min_val == max_val:
        return img
    return (img - min_val) / (max_val - min_val)

# Normalize all color images
recon_images_color = [normalize_image(img) for img in recon_images_color]

# Get output directory from output_path
output_path = config["postprocessing"]["output_path"]
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save results
# Create a figure with two rows: grayscale and color
fig, axs = plt.subplots(2, len(ks), figsize=(15, 8))

# First row: grayscale results
for i, (img, k) in enumerate(zip(recon_images_gray, ks)):
    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].set_title(f"Gray: Top {k} Coefs")
    axs[0, i].axis('off')

# Second row: color results
for i, (img, k) in enumerate(zip(recon_images_color, ks)):
    axs[1, i].imshow(img)
    axs[1, i].set_title(f"Color: Top {k} Coefs")
    axs[1, i].axis('off')

plt.tight_layout()

# Save the figure to the output directory
output_filename = os.path.join(output_dir, "wavelet_results_" + os.path.splitext(os.path.basename(img_path))[0] + ".png")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Results saved to {output_filename}")
