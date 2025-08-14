import torch
import torch.nn.functional as F
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any

class MseRenderer(VectorRenderer):
    """
    Renderer using MSE loss for optimization.
    This is the same as the base VectorRenderer implementation.
    """
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None, tile_size=32):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path, tile_size)
        
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor,
                    rendered_alpha: torch.Tensor = None) -> torch.Tensor:
        """
        Compute MSE loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3) or (H, W, 4) if it has an alpha channel
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            rendered_alpha: Optional alpha channel tensor (H, W) if available
            
        Returns:
            MSE loss value
        """
        # If target has an alpha channel, use it as a mask for loss calculation
        # In this case, we only compute loss for pixels where alpha > 0 (foreground pixels)
        # and include both RGB and alpha channel in the loss calculation
        if target.shape[2] == 4:
            assert rendered_alpha is not None, "Rendered alpha channel must be provided when target has an alpha channel."
            
            # Handle rendered_alpha shape: could be (H, W) or (H, W, 1)
            if rendered_alpha.dim() == 3 and rendered_alpha.shape[2] == 1:
                rendered_alpha = rendered_alpha.squeeze(-1)  # (H, W, 1) -> (H, W)

            # Extract alpha channel once and create mask
            target_alpha = target[:, :, 3]    # Shape: (H, W)
            alpha_mask = target_alpha > 0     # Shape: (H, W), boolean mask
            
            # Only compute loss for pixels where alpha > 0
            if alpha_mask.any():
                # Extract RGB channels
                target_rgb = target[:, :, :3]     # Shape: (H, W, 3)
                
                # Ensure consistent precision before masking to avoid unnecessary conversions
                if self.use_fp16:
                    if target_rgb.dtype == torch.float32:
                        rendered = rendered.float()
                        rendered_alpha = rendered_alpha.float()
                    elif rendered.dtype == torch.float16 and target_rgb.dtype != torch.float16:
                        target_rgb = target_rgb.half()
                        target_alpha = target_alpha.half()
                else:
                    rendered = rendered.float()
                    rendered_alpha = rendered_alpha.float()
                    target_rgb = target_rgb.float()
                    target_alpha = target_alpha.float()
                
                # Apply mask to all tensors
                rendered_masked = rendered[alpha_mask]              # Shape: (N_valid, 3)
                rendered_alpha_masked = rendered_alpha[alpha_mask]  # Shape: (N_valid,)
                target_rgb_masked = target_rgb[alpha_mask]          # Shape: (N_valid, 3)
                target_alpha_masked = target_alpha[alpha_mask]      # Shape: (N_valid,)
                
                # Combine RGB and alpha channels for single MSE computation
                # Concatenate along last dimension: RGB (3) + Alpha (1) = 4 channels
                rendered_combined = torch.cat([rendered_masked, rendered_alpha_masked.unsqueeze(-1)], dim=-1)  # (N_valid, 4)
                target_combined = torch.cat([target_rgb_masked, target_alpha_masked.unsqueeze(-1)], dim=-1)    # (N_valid, 4)
                
                return F.mse_loss(rendered_combined, target_combined)
            else:
                # If no valid pixels, return zero loss
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        else:
            # Original behavior for 3-channel targets
            # Ensure tensors are in consistent precision
            if self.use_fp16:
                if target.dtype == torch.float32:
                    rendered = rendered.float()
                elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                    target = target.half()
            else:
                rendered = rendered.float()
                target = target.float()


            return F.mse_loss(rendered, target)
    

    
    
    def compute_grayscale_loss(self, 
                          rendered_gray: torch.Tensor, 
                          target_gray: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between grayscale images.
        This focuses on structural similarity rather than color differences.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            
        Returns:
            MSE loss value computed on grayscale images
        """
        # Ensure tensors are in consistent precision
        if self.use_fp16:
            # If target is in FP32, convert rendered to FP32
            if target_gray.dtype == torch.float32:
                rendered_gray = rendered_gray.float()
            # If rendered is in FP16, convert target to FP16
            elif rendered_gray.dtype == torch.float16 and target_gray.dtype != torch.float16:
                target_gray = target_gray.half()
        else:
            # In FP32 mode, ensure everything is float32
            rendered_gray = rendered_gray.float()
            target_gray = target_gray.float()
        
        # Compute grayscale MSE loss
        grayscale_mse_loss = F.mse_loss(rendered_gray, target_gray)
        
        return grayscale_mse_loss
    
    def compute_combined_loss(self, 
                     rendered: torch.Tensor, 
                     target: torch.Tensor, 
                     x: torch.Tensor,
                     y: torch.Tensor,
                     r: torch.Tensor,
                     v: torch.Tensor,
                     theta: torch.Tensor,
                     c: torch.Tensor,
                     grayscale_weight: float = 0.7,
                     color_weight: float = 0.3,
                     use_gradient_loss: bool = False,
                     gradient_weight: float = 0.1,
                     use_cosine_similarity: bool = False,
                     use_canny_loss: bool = False,
                     canny_weight: float = 0.1) -> torch.Tensor:
        """
        Compute combined loss using both grayscale and color MSE losses.
        This balances structural similarity (grayscale) with color matching.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            x, y, r, v, theta, c: Current parameter values
            grayscale_weight: Weight for grayscale loss component (default: 0.7)
            color_weight: Weight for color loss component (default: 0.3)
            use_gradient_loss: If True, adds gradient-based loss for edge similarity
            gradient_weight: Weight for gradient loss component (default: 0.1)
            use_cosine_similarity: If True, uses cosine similarity for gradient loss
            use_canny_loss: If True, adds Canny edge-based loss for edge similarity
            canny_weight: Weight for Canny edge loss component (default: 0.1)
            
        Returns:
            Combined weighted loss value
        """
        # Convert RGB to grayscale using standard luminance weights
        # Y = 0.299*R + 0.587*G + 0.114*B
        rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114], 
                                         device=rendered.device, 
                                         dtype=rendered.dtype)
        
        # Convert rendered image to grayscale
        rendered_gray = torch.sum(rendered * rgb_to_gray_weights, dim=-1, keepdim=True)
        
        # Convert target image to grayscale
        target_gray = torch.sum(target * rgb_to_gray_weights, dim=-1, keepdim=True)
        
        # Compute grayscale-based structural loss
        grayscale_loss = self.compute_grayscale_loss(rendered_gray, target_gray)
        
        # Compute color-based loss
        color_loss = self.compute_loss(rendered, target, x, y, r, v, theta, c)
        
        # Start with weighted grayscale and color losses
        combined_loss = grayscale_weight * grayscale_loss + color_weight * color_loss
        
        # Add gradient-based loss with its own weight if requested
        if use_gradient_loss:
            gradient_loss = self._compute_gradient_loss(rendered_gray, target_gray, use_cosine_similarity)
            combined_loss = combined_loss + gradient_weight * gradient_loss
        
        # Add Canny edge-based loss with its own weight if requested
        if use_canny_loss:
            canny_loss = self._compute_canny_loss(rendered_gray, target_gray)
            combined_loss = combined_loss + canny_weight * canny_loss
        
        return combined_loss

    def _compute_gradient_loss(self, rendered_gray: torch.Tensor, target_gray: torch.Tensor, use_cosine_similarity: bool = False) -> torch.Tensor:
        """
        Compute gradient-based loss between grayscale images to focus on edge similarity.
        Uses Sobel operators to compute gradients in x and y directions.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            use_cosine_similarity: If True, uses cosine similarity loss on gradient vectors.
                                 If False, uses MSE loss on gradient magnitudes (absolute values).
            
        Returns:
            Gradient-based loss value
        """
        # Remove the channel dimension for gradient computation
        rendered_2d = rendered_gray.squeeze(-1)  # (H, W)
        target_2d = target_gray.squeeze(-1)      # (H, W)
        
        # Define Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        
        # Add batch and channel dimensions for conv2d
        rendered_batch = rendered_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_batch = target_2d.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
        
        # Compute gradients using Sobel operators
        rendered_grad_x = F.conv2d(rendered_batch, sobel_x, padding=1)
        rendered_grad_y = F.conv2d(rendered_batch, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_batch, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_batch, sobel_y, padding=1)
        
        if use_cosine_similarity:
            # Use cosine similarity loss on gradient vectors (preserves sign information)
            # Flatten gradients to compute per-pixel cosine similarity
            rendered_grad_flat = torch.stack([rendered_grad_x.flatten(), rendered_grad_y.flatten()], dim=0)  # (2, H*W)
            target_grad_flat = torch.stack([target_grad_x.flatten(), target_grad_y.flatten()], dim=0)  # (2, H*W)
            
            # Compute cosine similarity for each pixel's gradient vector
            # cosine_sim = (a · b) / (||a|| * ||b||)
            dot_product = torch.sum(rendered_grad_flat * target_grad_flat, dim=0)  # (H*W,)
            rendered_norm = torch.norm(rendered_grad_flat, dim=0) + 1e-8  # (H*W,)
            target_norm = torch.norm(target_grad_flat, dim=0) + 1e-8  # (H*W,)
            
            cosine_sim = dot_product / (rendered_norm * target_norm)  # (H*W,)
            
            # Convert cosine similarity to loss (1 - cosine_sim), then take mean
            # Cosine similarity ranges from -1 to 1, so (1 - cosine_sim) ranges from 0 to 2
            gradient_loss = torch.mean(1.0 - cosine_sim)
        else:
            # Use MSE loss on gradient magnitudes (absolute values)
            rendered_grad_mag = torch.sqrt(rendered_grad_x**2 + rendered_grad_y**2 + 1e-8)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
            gradient_loss = F.mse_loss(rendered_grad_mag, target_grad_mag)
    
        return gradient_loss

    def _compute_canny_loss(self, rendered_gray: torch.Tensor, target_gray: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified Canny-inspired edge loss between grayscale images.
        Uses differentiable operations to maintain gradient flow during optimization.
        
        Args:
            rendered_gray: Rendered grayscale image tensor (H, W, 1)
            target_gray: Target grayscale image tensor (H, W, 1)
            
        Returns:
            Canny-inspired edge loss value
        """
        # Remove the channel dimension for edge detection
        rendered_2d = rendered_gray.squeeze(-1)  # (H, W)
        target_2d = target_gray.squeeze(-1)      # (H, W)
        
        # Add batch and channel dimensions for processing
        rendered_batch = rendered_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target_batch = target_2d.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
        
        # Step 1: Apply Gaussian smoothing (3x3 kernel for efficiency)
        gaussian_kernel = self._get_gaussian_kernel(3, 0.8, rendered_2d.device, rendered_2d.dtype)
        rendered_smooth = F.conv2d(rendered_batch, gaussian_kernel, padding=1)
        target_smooth = F.conv2d(target_batch, gaussian_kernel, padding=1)
        
        # Step 2: Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=rendered_2d.dtype, device=rendered_2d.device).unsqueeze(0).unsqueeze(0)
        
        # Compute gradients
        rendered_grad_x = F.conv2d(rendered_smooth, sobel_x, padding=1)
        rendered_grad_y = F.conv2d(rendered_smooth, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_smooth, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_smooth, sobel_y, padding=1)
        
        # Step 3: Compute gradient magnitude
        rendered_grad_mag = torch.sqrt(rendered_grad_x**2 + rendered_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # Step 4: Differentiable edge detection using soft thresholding
        # Normalize gradient magnitudes to [0, 1] range
        rendered_norm = rendered_grad_mag / (rendered_grad_mag.max() + 1e-8)
        target_norm = target_grad_mag / (target_grad_mag.max() + 1e-8)
        
        # Apply soft thresholding using sigmoid function for differentiability
        # This replaces the hard thresholding in traditional Canny
        # Using fixed thresholds: low=0.1, high=0.2 (commonly used values)
        high_threshold = 0.2
        steepness = 10.0  # Controls the steepness of the sigmoid
        rendered_edges = torch.sigmoid(steepness * (rendered_norm - high_threshold))
        target_edges = torch.sigmoid(steepness * (target_norm - high_threshold))
        
        # Compute MSE loss between edge maps
        canny_loss = F.mse_loss(rendered_edges, target_edges)
        
        return canny_loss

    
    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate a 2D Gaussian kernel for smoothing.
        """
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian values
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        
