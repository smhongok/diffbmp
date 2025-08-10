#!/usr/bin/env python3
"""
GIF Canny Edge Visualization Script

This script extracts and visualizes the Canny edge map for each frame of a GIF file
using the same implementation as the _compute_canny_loss method in MseRenderer.

Usage:
    python visualize_gif_canny_edges.py --gif_path images/animation.gif
    python visualize_gif_canny_edges.py --gif_path images/animation.gif --output_dir gif_canny_output
    python visualize_gif_canny_edges.py --gif_path images/animation.gif --save_gif
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List, Tuple


class GifCannyEdgeVisualizer:
    """Visualizes Canny edge maps for each frame of a GIF using the same implementation as MseRenderer."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_gif_frames(self, gif_path: str, target_width: int = 256) -> List[torch.Tensor]:
        """Load and preprocess all frames from a GIF file."""
        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF not found: {gif_path}")
        
        # Load GIF
        gif = Image.open(gif_path)
        frames = []
        
        try:
            frame_count = 0
            while True:
                # Convert frame to RGB (in case it's in palette mode)
                frame = gif.convert('RGB')
                print(f"Frame {frame_count}: Original size: {frame.size}")
                
                # Resize to target width while maintaining aspect ratio
                aspect_ratio = frame.height / frame.width
                target_height = int(target_width * aspect_ratio)
                frame = frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
                print(f"Frame {frame_count}: Resized size: {frame.size}")
                
                # Convert to tensor and normalize to [0, 1]
                frame_array = np.array(frame).astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_array).to(self.device)
                frames.append(frame_tensor)
                
                frame_count += 1
                gif.seek(gif.tell() + 1)
                
        except EOFError:
            pass  # End of GIF frames
        
        print(f"Loaded {len(frames)} frames from GIF")
        return frames
    
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
        gaussian = gaussian / gaussian.sum()
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)
    
    def non_maximum_suppression(self, magnitude: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression to thin edges."""
        H, W = magnitude.shape
        suppressed = torch.zeros_like(magnitude)
        
        # Convert direction to degrees and normalize to [0, 180)
        direction_deg = (direction * 180 / torch.pi) % 180
        
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                angle = direction_deg[i, j].item()
                
                # Determine neighboring pixels based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                    # Horizontal direction
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= angle < 67.5:
                    # Diagonal direction (/)
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= angle < 112.5:
                    # Vertical direction
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= angle < 157.5
                    # Diagonal direction (\)
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                
                # Keep pixel if it's a local maximum
                if magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed
    
    def double_threshold(self, magnitude: torch.Tensor, low_threshold: float, high_threshold: float) -> torch.Tensor:
        """Apply double thresholding to classify edges."""
        strong_edges = magnitude >= high_threshold
        weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)
        
        # For simplicity, we'll just use strong edges + weak edges
        # A full implementation would do edge tracking by hysteresis
        edges = strong_edges.float() + weak_edges.float() * 0.5
        
        return edges
    
    def compute_canny_edges(self, image: torch.Tensor, low_threshold: float = 0.1, high_threshold: float = 0.2) -> torch.Tensor:
        """Compute Canny edge map using the same method as _compute_canny_loss."""
        # Convert to grayscale
        if image.dim() == 3 and image.shape[-1] == 3:
            gray_image = self.rgb_to_grayscale(image).squeeze(-1)  # Remove channel dimension
        else:
            gray_image = image.squeeze() if image.dim() > 2 else image
        
        # Add batch and channel dimensions for conv2d: (1, 1, H, W)
        gray_batch = gray_image.unsqueeze(0).unsqueeze(0)
        
        # Step 1: Apply Gaussian smoothing (3x3 kernel for efficiency)
        gaussian_kernel = self.get_gaussian_kernel(3, 0.8, self.device, gray_image.dtype)
        smoothed = F.conv2d(gray_batch, gaussian_kernel, padding=1)
        
        # Step 2: Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=gray_image.dtype, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=gray_image.dtype, device=self.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(smoothed, sobel_x, padding=1)
        grad_y = F.conv2d(smoothed, sobel_y, padding=1)
        
        # Step 3: Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Step 4: Differentiable edge detection using soft thresholding
        # Normalize gradient magnitudes to [0, 1] range
        grad_norm = grad_magnitude / (grad_magnitude.max() + 1e-8)
        
        # Apply soft thresholding using sigmoid function for differentiability
        # This replaces the hard thresholding in traditional Canny
        steepness = 10.0  # Controls the steepness of the sigmoid
        edges = torch.sigmoid(steepness * (grad_norm - high_threshold))
        
        # Remove batch and channel dimensions and return
        return edges.squeeze()
    
    def visualize_frame_canny_edges(self, frames: List[torch.Tensor], 
                                  low_threshold: float = 0.1, 
                                  high_threshold: float = 0.2,
                                  output_dir: str = None,
                                  save_gif: bool = False) -> List[np.ndarray]:
        """Visualize Canny edge maps for all frames."""
        edge_frames = []
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing {len(frames)} frames...")
        print(f"Canny parameters: low_threshold={low_threshold}, high_threshold={high_threshold}")
        
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            
            # Compute Canny edges
            edges = self.compute_canny_edges(frame, low_threshold, high_threshold)
            
            # Convert to numpy for visualization
            frame_np = frame.cpu().numpy()
            edges_np = edges.cpu().numpy()
            
            # Store edge frame for GIF creation
            edge_frames.append((edges_np * 255).astype(np.uint8))
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original frame
            axes[0].imshow(frame_np)
            axes[0].set_title(f'Original Frame {i+1}')
            axes[0].axis('off')
            
            # Grayscale frame
            gray_frame = self.rgb_to_grayscale(frame).squeeze().cpu().numpy()
            axes[1].imshow(gray_frame, cmap='gray')
            axes[1].set_title(f'Grayscale Frame {i+1}')
            axes[1].axis('off')
            
            # Canny edges
            axes[2].imshow(edges_np, cmap='gray')
            axes[2].set_title(f'Canny Edges Frame {i+1}')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save individual frame visualization
            if output_dir:
                frame_output_path = os.path.join(output_dir, f'frame_{i+1:03d}_canny.png')
                plt.savefig(frame_output_path, dpi=150, bbox_inches='tight')
                print(f"Saved frame visualization: {frame_output_path}")
            
            plt.show()
            plt.close()  # Close figure to prevent memory accumulation
        
        # Create and save GIF of edge maps if requested
        if save_gif and edge_frames:
            self.save_edge_gif(edge_frames, output_dir or '.', 'canny_edges.gif')
        
        return edge_frames
    
    def save_edge_gif(self, edge_frames: List[np.ndarray], output_dir: str, filename: str):
        """Save edge frames as an animated GIF."""
        # Convert numpy arrays to PIL Images
        pil_frames = []
        for edge_frame in edge_frames:
            # Convert to PIL Image
            pil_frame = Image.fromarray(edge_frame, mode='L')  # 'L' for grayscale
            pil_frames.append(pil_frame)
        
        # Save as animated GIF
        output_path = os.path.join(output_dir, filename)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=200,  # 200ms per frame (5 FPS)
            loop=0  # Infinite loop
        )
        print(f"Saved Canny edge GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Canny edge maps for GIF frames')
    parser.add_argument('--gif_path', type=str, required=True,
                       help='Path to the input GIF file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output images (optional)')
    parser.add_argument('--save_gif', action='store_true',
                       help='Save edge maps as an animated GIF')
    parser.add_argument('--target_width', type=int, default=256,
                       help='Target width for frame resizing (default: 256)')
    parser.add_argument('--low_threshold', type=float, default=0.1,
                       help='Canny low threshold (default: 0.1)')
    parser.add_argument('--high_threshold', type=float, default=0.2,
                       help='Canny high threshold (default: 0.2)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = GifCannyEdgeVisualizer()
    
    try:
        # Load GIF frames
        print(f"Loading GIF: {args.gif_path}")
        frames = visualizer.load_gif_frames(args.gif_path, args.target_width)
        
        # Visualize Canny edges for all frames
        edge_frames = visualizer.visualize_frame_canny_edges(
            frames, 
            args.low_threshold, 
            args.high_threshold,
            args.output_dir,
            args.save_gif
        )
        
        print(f"\nProcessing complete! Processed {len(edge_frames)} frames.")
        if args.output_dir:
            print(f"Individual frame visualizations saved to: {args.output_dir}")
        if args.save_gif:
            print(f"Canny edge GIF saved to: {args.output_dir or '.'}/canny_edges.gif")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
