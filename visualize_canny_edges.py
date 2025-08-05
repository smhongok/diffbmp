#!/usr/bin/env python3
"""
Canny Edge Visualization Script

This script extracts and visualizes the Canny edge map for image2 using the same
implementation as the _compute_canny_loss method in MseRenderer.

Usage:
    python visualize_canny_edges.py --config configs/default_xy_dynamics.json
    python visualize_canny_edges.py --image_path images/nature/melon.jpg
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


class CannyEdgeVisualizer:
    """Visualizes Canny edge maps using the same implementation as MseRenderer."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_image(self, image_path: str, target_width: int = 256) -> torch.Tensor:
        """Load and preprocess image similar to the main pipeline."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"Original image size: {image.size}")
        
        # Resize to target width while maintaining aspect ratio
        aspect_ratio = image.height / image.width
        target_height = int(target_width * aspect_ratio)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        print(f"Resized image size: {image.size}")
        
        # Convert to tensor and normalize to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).to(self.device)
        
        return image_tensor
    
    def rgb_to_grayscale(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale using standard luminance weights."""
        # Y = 0.299*R + 0.587*G + 0.114*B
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], 
                                         device=rgb_image.device, 
                                         dtype=rgb_image.dtype)
        grayscale = torch.sum(rgb_image * rgb_to_gray_weights, dim=-1, keepdim=True)
        return grayscale
    
    def get_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate a 2D Gaussian kernel for smoothing."""
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian values
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    def compute_canny_edges(self, grayscale_image: torch.Tensor, 
                           low_threshold: float = 0.1, 
                           high_threshold: float = 0.2) -> dict:
        """
        Compute Canny edge map using the same implementation as _compute_canny_loss.
        
        Returns:
            dict: Contains intermediate results for visualization
        """
        # Remove the channel dimension for edge detection
        image_2d = grayscale_image.squeeze(-1)  # (H, W)
        
        # Add batch and channel dimensions for processing
        image_batch = image_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Step 1: Apply Gaussian smoothing (3x3 kernel for efficiency)
        gaussian_kernel = self.get_gaussian_kernel(3, 0.8, image_2d.device, image_2d.dtype)
        image_smooth = F.conv2d(image_batch, gaussian_kernel, padding=1)
        
        # Step 2: Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=image_2d.dtype, device=image_2d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=image_2d.dtype, device=image_2d.device).unsqueeze(0).unsqueeze(0)
        
        # Compute gradients
        grad_x = F.conv2d(image_smooth, sobel_x, padding=1)
        grad_y = F.conv2d(image_smooth, sobel_y, padding=1)
        
        # Step 3: Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Step 4: Differentiable edge detection using soft thresholding
        # Normalize gradient magnitudes to [0, 1] range
        grad_norm = grad_mag / (grad_mag.max() + 1e-8)
        
        # Apply soft thresholding using sigmoid function for differentiability
        steepness = 10.0  # Controls the steepness of the sigmoid
        edges = torch.sigmoid(steepness * (grad_norm - high_threshold))
        
        return {
            'original': grayscale_image.squeeze(-1).cpu().numpy(),
            'smoothed': image_smooth.squeeze().cpu().numpy(),
            'grad_x': grad_x.squeeze().cpu().numpy(),
            'grad_y': grad_y.squeeze().cpu().numpy(),
            'grad_magnitude': grad_mag.squeeze().cpu().numpy(),
            'grad_normalized': grad_norm.squeeze().cpu().numpy(),
            'edges': edges.squeeze().cpu().numpy(),
            'low_threshold': low_threshold,
            'high_threshold': high_threshold
        }
    
    def visualize_results(self, results: dict, save_path: str = None):
        """Create a comprehensive visualization of the Canny edge detection process."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Canny Edge Detection Process Visualization', fontsize=16)
        
        # Original grayscale image
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original Grayscale')
        axes[0, 0].axis('off')
        
        # Gaussian smoothed image
        axes[0, 1].imshow(results['smoothed'], cmap='gray')
        axes[0, 1].set_title('Gaussian Smoothed')
        axes[0, 1].axis('off')
        
        # Gradient X
        axes[0, 2].imshow(results['grad_x'], cmap='gray')
        axes[0, 2].set_title('Gradient X (Sobel)')
        axes[0, 2].axis('off')
        
        # Gradient Y
        axes[0, 3].imshow(results['grad_y'], cmap='gray')
        axes[0, 3].set_title('Gradient Y (Sobel)')
        axes[0, 3].axis('off')
        
        # Gradient magnitude
        axes[1, 0].imshow(results['grad_magnitude'], cmap='hot')
        axes[1, 0].set_title('Gradient Magnitude')
        axes[1, 0].axis('off')
        
        # Normalized gradient magnitude
        axes[1, 1].imshow(results['grad_normalized'], cmap='hot')
        axes[1, 1].set_title('Normalized Gradient')
        axes[1, 1].axis('off')
        
        # Final edge map
        axes[1, 2].imshow(results['edges'], cmap='gray')
        axes[1, 2].set_title(f'Canny Edges\n(threshold: {results["high_threshold"]})')
        axes[1, 2].axis('off')
        
        # Edge map with colormap for better visibility
        im = axes[1, 3].imshow(results['edges'], cmap='plasma')
        axes[1, 3].set_title('Edges (Colored)')
        axes[1, 3].axis('off')
        plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def process_from_config(self, config_path: str):
        """Process image2 from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get image paths and parameters
        img_paths = config['preprocessing']['img_paths']
        if len(img_paths) < 2:
            raise ValueError("Configuration must contain at least 2 image paths for XY dynamics mode")
        
        image2_path = img_paths[1]  # Second image
        target_width = config['preprocessing']['final_width'][0]
        
        # Get Canny parameters from xy_optimization config
        xy_opt_conf = config['xy_dynamics']['xy_optimization']
        low_threshold = xy_opt_conf.get('canny_low_threshold', 0.1)
        high_threshold = xy_opt_conf.get('canny_high_threshold', 0.2)
        
        print(f"Processing image2: {image2_path}")
        print(f"Target width: {target_width}")
        print(f"Canny thresholds: low={low_threshold}, high={high_threshold}")
        
        return self.process_image(image2_path, target_width, low_threshold, high_threshold)
    
    def process_image(self, image_path: str, target_width: int = 256, 
                     low_threshold: float = 0.1, high_threshold: float = 0.2):
        """Process a single image and visualize its Canny edges."""
        # Load and preprocess image
        rgb_image = self.load_image(image_path, target_width)
        grayscale_image = self.rgb_to_grayscale(rgb_image)
        
        # Compute Canny edges
        results = self.compute_canny_edges(grayscale_image, low_threshold, high_threshold)
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"canny_visualization_{base_name}.png"
        
        # Visualize results
        self.visualize_results(results, output_path)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Visualize Canny edge detection for image2')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--image_path', type=str, help='Direct path to image file')
    parser.add_argument('--width', type=int, default=256, help='Target width for resizing')
    parser.add_argument('--low_threshold', type=float, default=0.1, help='Low threshold for Canny')
    parser.add_argument('--high_threshold', type=float, default=0.2, help='High threshold for Canny')
    
    args = parser.parse_args()
    
    visualizer = CannyEdgeVisualizer()
    
    if args.config:
        # Process from configuration file
        visualizer.process_from_config(args.config)
    elif args.image_path:
        # Process single image
        visualizer.process_image(args.image_path, args.width, args.low_threshold, args.high_threshold)
    else:
        print("Please provide either --config or --image_path argument")
        parser.print_help()


if __name__ == "__main__":
    main()
