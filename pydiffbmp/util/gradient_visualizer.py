"""
Gradient Visualization Utility Class

This module provides a reusable class for visualizing per-pixel gradient directions
for primitive-based rendering systems. It supports different color spectrums and
can be used to analyze gradient patterns for different sets of primitives.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle
import os
from typing import Optional, Union, List, Tuple, Dict, Any


class GradientVisualizer:
    """
    A class for visualizing per-pixel gradient directions for primitive-based rendering.
    
    This class stores target image and save path configuration, then provides methods
    to visualize gradients for different sets of primitives with configurable color
    spectrums and visualization parameters.
    """
    
    def __init__(self, 
                 target_image: torch.Tensor,
                 save_path: str,
                 color_spectrum: str = "full",
                 background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 primitive_radius_multiplier: float = 1.5,
                 gradient_threshold: float = 1e-14,
                 center_dot_radius: int = 3,
                 enable_logging: bool = True):
        """
        Initialize the gradient visualizer.
        
        Args:
            target_image: Target image tensor [H, W, 3]
            save_path: Base path for saving visualizations (without extension)
            color_spectrum: Color spectrum type ("full", "warm", "cool", "custom")
            background_color: RGB background color (default: white)
            primitive_radius_multiplier: Multiplier for primitive visualization radius
            gradient_threshold: Threshold for considering gradients as non-zero
            center_dot_radius: Radius of center dots marking primitive positions
            enable_logging: Whether to enable detailed logging
        """
        self.target_image = target_image
        self.save_path = save_path
        self.color_spectrum = color_spectrum
        self.background_color = background_color
        self.primitive_radius_multiplier = primitive_radius_multiplier
        self.gradient_threshold = gradient_threshold
        self.center_dot_radius = center_dot_radius
        self.enable_logging = enable_logging
        
        # Extract image dimensions
        self.H, self.W = target_image.shape[:2]
        
        # Validate color spectrum
        valid_spectrums = ["full", "warm", "cool", "custom"]
        if color_spectrum not in valid_spectrums:
            raise ValueError(f"color_spectrum must be one of {valid_spectrums}, got {color_spectrum}")
        
        # Create save directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if self.enable_logging:
            print(f"[GradientVisualizer] Initialized with target image {self.H}x{self.W}, "
                  f"color_spectrum='{color_spectrum}', save_path='{save_path}'")
    
    def _get_color_spectrum_hue_range(self) -> Tuple[float, float]:
        """Get hue range for the selected color spectrum."""
        if self.color_spectrum == "full":
            return (0.0, 1.0)  # Full spectrum: 0° to 360°
        elif self.color_spectrum == "warm":
            return (0.0, 0.25)  # Warm colors: 0° to 90° (red to yellow)
        elif self.color_spectrum == "cool":
            return (0.5, 0.75)  # Cool colors: 180° to 270° (cyan to blue)
        else:  # custom
            return (0.0, 1.0)  # Default to full spectrum for custom
    
    def _render_and_compute_pixel_losses(self,
                                        renderer,
                                        x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                        v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                        primitive_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render the current state and compute per-pixel losses against target image.
        
        Args:
            renderer: Renderer object with render_from_params method
            x, y, r, v, theta, c: Primitive parameters
            primitive_indices: Indices of primitives to analyze
            
        Returns:
            Tuple of (rendered_image, pixel_losses)
        """
        # Ensure x, y parameters require gradients
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        
        # Render current state
        rendered = renderer.render_from_params(x, y, r, v, theta, c)
        
        # Compute pixel-wise loss (no reduction)
        pixel_losses = (rendered - self.target_image).pow(2).mean(dim=2)  # [H, W]
        
        return rendered, pixel_losses
    
    def _process_primitive_gradients(self,
                                   renderer,
                                   x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                                   v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                                   pixel_losses: torch.Tensor,
                                   primitive_indices: torch.Tensor,
                                   canvas: np.ndarray,
                                   gradient_mask: np.ndarray,
                                   center_dot_color: Tuple[float, float, float]) -> None:
        """
        Process gradients for a set of primitives and update the visualization canvas.
        
        Args:
            renderer: Renderer object
            x, y, r, v, theta, c: Primitive parameters
            pixel_losses: Pre-computed pixel losses [H, W]
            primitive_indices: Indices of primitives to process
            canvas: Visualization canvas to update [H, W, 3]
            gradient_mask: Mask tracking processed pixels [H, W]
            center_dot_color: RGB color for center dots
        """
        hue_min, hue_max = self._get_color_spectrum_hue_range()
        
        for idx, prim_idx in enumerate(primitive_indices):
            prim_idx = int(prim_idx.item()) if torch.is_tensor(prim_idx) else int(prim_idx)
            
            # Get primitive parameters
            prim_x = float(x[prim_idx].item())
            prim_y = float(y[prim_idx].item())
            prim_r = float(r[prim_idx].item())
            
            # Define circular region
            radius = self.primitive_radius_multiplier * prim_r
            
            if self.enable_logging:
                print(f"[GradientVisualizer] Processing primitive {prim_idx}: "
                      f"center=({prim_x:.1f}, {prim_y:.1f}), radius={radius:.1f}")
            
            # Find pixels within the circular region
            y_coords, x_coords = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
            distances = np.sqrt((x_coords - prim_x)**2 + (y_coords - prim_y)**2)
            circle_mask = distances <= radius
            
            # Get pixel coordinates within the circle
            circle_pixels = np.where(circle_mask)
            num_pixels = len(circle_pixels[0])
            
            if num_pixels == 0:
                if self.enable_logging:
                    print(f"[GradientVisualizer] No pixels found for primitive {prim_idx}")
                continue
            
            if self.enable_logging:
                print(f"[GradientVisualizer] Processing {num_pixels} pixels for primitive {prim_idx}")
            
            # Compute gradients for pixels in this circle
            gradient_directions = np.zeros((num_pixels, 2))  # [num_pixels, 2] for x,y gradients
            
            for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                pixel_loss = pixel_losses[i, j]
                
                if pixel_loss.requires_grad:
                    # Compute gradients w.r.t. x, y for this pixel
                    grads = torch.autograd.grad(
                        outputs=pixel_loss,
                        inputs=[x, y],
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Extract gradients for the current primitive
                    if grads[0] is not None and grads[1] is not None:
                        grad_x = float(grads[0][prim_idx].item())
                        grad_y = float(grads[1][prim_idx].item())
                        gradient_directions[pixel_idx] = [grad_x, grad_y]
            
            # Compute magnitude statistics for adaptive normalization
            magnitudes = np.sqrt(gradient_directions[:, 0]**2 + gradient_directions[:, 1]**2)
            non_zero_magnitudes = magnitudes[magnitudes > self.gradient_threshold]
            
            if len(non_zero_magnitudes) > 0:
                # Adaptive normalization based on this primitive's gradient range
                mag_min = np.min(non_zero_magnitudes)
                mag_max = np.max(non_zero_magnitudes)
                mag_median = np.median(non_zero_magnitudes)
                
                if self.enable_logging:
                    print(f"[GradientVisualizer] Primitive {prim_idx} gradient stats: "
                          f"min={mag_min:.2e}, max={mag_max:.2e}, median={mag_median:.2e}")
                
                # Use logarithmic scaling to better visualize small gradients
                log_min = np.log10(mag_min + self.gradient_threshold)
                log_max = np.log10(mag_max + self.gradient_threshold)
                log_range = log_max - log_min if log_max > log_min else 1.0
                
                # Convert gradient directions to colors and apply to canvas
                for pixel_idx, (i, j) in enumerate(zip(circle_pixels[0], circle_pixels[1])):
                    grad_x, grad_y = gradient_directions[pixel_idx]
                    
                    # Compute gradient magnitude and direction
                    magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    if magnitude > self.gradient_threshold:
                        # Compute angle in radians, then convert to hue range
                        angle = np.arctan2(grad_y, grad_x)  # [-π, π]
                        normalized_angle = (angle + np.pi) / (2 * np.pi)  # [0, 1]
                        
                        # Map to selected color spectrum
                        hue = hue_min + normalized_angle * (hue_max - hue_min)
                        
                        # Enhanced saturation calculation using logarithmic scaling
                        if log_range > 0:
                            log_magnitude = np.log10(magnitude + self.gradient_threshold)
                            normalized_log_mag = (log_magnitude - log_min) / log_range
                            saturation = np.clip(normalized_log_mag, 0.1, 1.0)
                        else:
                            saturation = 0.5
                        
                        # Adaptive value (brightness) based on magnitude relative to median
                        if magnitude >= mag_median:
                            value = 0.9  # High brightness for above-median gradients
                        else:
                            relative_mag = magnitude / mag_median
                            value = 0.4 + 0.4 * relative_mag
                        
                        # Convert HSV to RGB
                        rgb_color = hsv_to_rgb([hue, saturation, value])
                        
                        # Set color on canvas
                        canvas[i, j] = rgb_color
                        gradient_mask[i, j] = True
                    else:
                        # For zero gradients, use light gray
                        canvas[i, j] = [0.95, 0.95, 0.95]
                        gradient_mask[i, j] = True
            else:
                if self.enable_logging:
                    print(f"[GradientVisualizer] Primitive {prim_idx} has no significant gradients")
        
        # Add primitive centers as dots
        for prim_idx in primitive_indices:
            prim_idx = int(prim_idx.item()) if torch.is_tensor(prim_idx) else int(prim_idx)
            prim_x = int(x[prim_idx].item())
            prim_y = int(y[prim_idx].item())
            
            # Draw center dot
            if 0 <= prim_x < self.W and 0 <= prim_y < self.H:
                for dy in range(-self.center_dot_radius, self.center_dot_radius + 1):
                    for dx in range(-self.center_dot_radius, self.center_dot_radius + 1):
                        if dx*dx + dy*dy <= self.center_dot_radius*self.center_dot_radius:
                            nx, ny = prim_x + dx, prim_y + dy
                            if 0 <= nx < self.W and 0 <= ny < self.H:
                                canvas[ny, nx] = center_dot_color
    
    def _save_visualization(self, 
                          canvas: np.ndarray, 
                          primitive_indices: torch.Tensor,
                          suffix: str,
                          title_prefix: str,
                          center_dot_description: str) -> str:
        """
        Save a visualization canvas to file.
        
        Args:
            canvas: Visualization canvas [H, W, 3]
            primitive_indices: Indices of visualized primitives
            suffix: Filename suffix
            title_prefix: Title prefix for the plot
            center_dot_description: Description of center dots
            
        Returns:
            Path where the visualization was saved
        """
        # Create filename
        save_dir = os.path.dirname(self.save_path)
        filename = os.path.basename(self.save_path)
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        
        output_path = os.path.join(save_dir, f"{name}_{suffix}{ext}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.imshow(canvas)
        plt.title(f'{title_prefix} ({len(primitive_indices)} primitives)\n'
                 f'{center_dot_description}, Color = gradient direction & magnitude')
        plt.axis('off')
        
        # Add color wheel for gradient directions
        ax = plt.gca()
        wheel_center = (self.W * 0.9, self.H * 0.1)
        wheel_radius = min(self.W, self.H) * 0.05
        
        hue_min, hue_max = self._get_color_spectrum_hue_range()
        
        # Create color wheel showing the used spectrum
        angles = np.linspace(0, 2*np.pi, 360)
        for i, angle in enumerate(angles):
            normalized_angle = angle / (2 * np.pi)
            hue = hue_min + normalized_angle * (hue_max - hue_min)
            color = hsv_to_rgb([hue, 1.0, 0.9])
            
            x_pos = wheel_center[0] + wheel_radius * np.cos(angle)
            y_pos = wheel_center[1] + wheel_radius * np.sin(angle)
            
            circle = Circle((x_pos, y_pos), wheel_radius/20, color=color)
            ax.add_patch(circle)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.enable_logging:
            print(f"[GradientVisualizer] Saved {suffix} visualization to {output_path}")
        
        return output_path
    
    def visualize_gradients(self,
                          renderer,
                          x: torch.Tensor, y: torch.Tensor, r: torch.Tensor,
                          v: torch.Tensor, theta: torch.Tensor, c: torch.Tensor,
                          primitive_indices: torch.Tensor,
                          suffix: str = "gradients",
                          title_prefix: str = "Gradient Visualization",
                          center_dot_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[torch.Tensor, str]:
        """
        Visualize per-pixel gradients for a set of primitives.
        
        Args:
            renderer: Renderer object with render_from_params method
            x, y, r, v, theta, c: Primitive parameters
            primitive_indices: Indices of primitives to visualize
            suffix: Filename suffix for saved visualization
            title_prefix: Title prefix for the plot
            center_dot_color: RGB color for center dots marking primitive positions
            
        Returns:
            Tuple of (visualization_tensor, saved_path)
        """
        if self.enable_logging:
            print(f"[GradientVisualizer] Visualizing gradients for {len(primitive_indices)} primitives")
        
        # Render and compute pixel losses
        rendered, pixel_losses = self._render_and_compute_pixel_losses(
            renderer, x, y, r, v, theta, c, primitive_indices)
        
        # Create visualization canvas
        canvas = np.ones((self.H, self.W, 3), dtype=np.float32) * np.array(self.background_color)
        gradient_mask = np.zeros((self.H, self.W), dtype=bool)
        
        # Process primitive gradients
        self._process_primitive_gradients(
            renderer, x, y, r, v, theta, c, pixel_losses, primitive_indices,
            canvas, gradient_mask, center_dot_color)
        
        # Convert to tensor
        vis_tensor = torch.from_numpy(canvas).to(rendered.device)
        
        # Save visualization
        saved_path = self._save_visualization(
            canvas, primitive_indices, suffix, title_prefix,
            f"Dots = primitive centers")
        
        if self.enable_logging:
            print(f"[GradientVisualizer] Completed visualization for {len(primitive_indices)} primitives")
        
        return vis_tensor, saved_path
    
    def set_color_spectrum(self, color_spectrum: str) -> None:
        """
        Update the color spectrum for future visualizations.
        
        Args:
            color_spectrum: New color spectrum ("full", "warm", "cool", "custom")
        """
        valid_spectrums = ["full", "warm", "cool", "custom"]
        if color_spectrum not in valid_spectrums:
            raise ValueError(f"color_spectrum must be one of {valid_spectrums}, got {color_spectrum}")
        
        self.color_spectrum = color_spectrum
        if self.enable_logging:
            print(f"[GradientVisualizer] Updated color spectrum to '{color_spectrum}'")
    
    def set_custom_hue_range(self, hue_min: float, hue_max: float) -> None:
        """
        Set custom hue range for "custom" color spectrum.
        
        Args:
            hue_min: Minimum hue value [0, 1]
            hue_max: Maximum hue value [0, 1]
        """
        if not (0 <= hue_min <= 1 and 0 <= hue_max <= 1):
            raise ValueError("Hue values must be in range [0, 1]")
        
        self.custom_hue_range = (hue_min, hue_max)
        if self.enable_logging:
            print(f"[GradientVisualizer] Set custom hue range: [{hue_min:.2f}, {hue_max:.2f}]")
    
    def _get_color_spectrum_hue_range(self) -> Tuple[float, float]:
        """Get hue range for the selected color spectrum."""
        if self.color_spectrum == "full":
            return (0.0, 1.0)  # Full spectrum: 0° to 360°
        elif self.color_spectrum == "warm":
            return (0.0, 0.25)  # Warm colors: 0° to 90° (red to yellow)
        elif self.color_spectrum == "cool":
            return (0.5, 0.75)  # Cool colors: 180° to 270° (cyan to blue)
        elif self.color_spectrum == "custom" and hasattr(self, 'custom_hue_range'):
            return self.custom_hue_range
        else:  # custom without range set, default to full
            return (0.0, 1.0)
