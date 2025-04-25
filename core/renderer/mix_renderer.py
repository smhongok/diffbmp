import torch
import torch.nn.functional as F
import numpy as np
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any

class MixRenderer(VectorRenderer):
    """
    Renderer using a combination of shape alignment loss (Mask IoU/Dice) for geometric parameters
    and masked L1 loss for color/alpha parameters.
    """
    def compute_shape_alignment_loss(self, 
                                   pred_masks: torch.Tensor, 
                                   target_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target masks.
        
        Args:
            pred_masks: Predicted binary masks (B, H, W)
            target_masks: Target binary mask (H, W)
            
        Returns:
            IoU loss value (ℓmask = 1 - IoU)
        """
        # Expand target mask to match batch dimension
        target_masks = target_masks.unsqueeze(0).expand_as(pred_masks)
        
        # Compute intersection and union
        mul_masks = pred_masks * target_masks
        intersection = torch.sum(mul_masks, dim=(1, 2))
        union = torch.sum(pred_masks + target_masks - mul_masks, dim=(1, 2)).float() + 1e-8
        
        # Compute IoU
        iou = intersection / union
        
        # Return mean IoU loss
        return 1.0 - torch.mean(iou)

    def compute_color_alpha_loss(self, 
                               rendered: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
        """
        Compute masked L1 loss for color and alpha values.
        Only compute loss in regions where the target mask is active.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            masks: Target binary mask (H, W)
            
        Returns:
            Masked L1 loss value (ℓcol)
        """
        # Get target mask (Mgt)
        target_mask = (masks > 0.1).float()
        
        # Compute absolute difference
        diff = torch.abs(rendered - target)
        
        # Compute sum of absolute differences in masked region
        masked_diff = diff * target_mask.unsqueeze(-1)
        total_diff = torch.sum(masked_diff)
        
        # Normalize by number of pixels in target mask
        num_mask_pixels = torch.sum(target_mask) + 1e-8
        return total_diff / num_mask_pixels

    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    cached_masks: torch.Tensor,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute shape and color losses separately.
        Shape loss (ℓmask) affects x, y, r, theta parameters.
        Color loss (ℓcol) affects c, v parameters.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            Tuple of (total_loss, shape_loss, color_loss)
        """
        # Extract target mask from target image (assuming first channel represents mask)
        target_mask = target[..., 0]
        
        # Compute shape alignment loss (affects x, y, r, theta)
        shape_loss = self.compute_shape_alignment_loss(cached_masks, target_mask)
        
        # Compute color/alpha loss (affects c, v)
        color_loss = self.compute_color_alpha_loss(rendered, target, target_mask)
        
        # Combine losses with weighting
        shape_weight = 0.7
        color_weight = 0.3
        total_loss = shape_weight * shape_loss + color_weight * color_loss
        
        return total_loss, shape_loss, color_loss

    def optimize_parameters(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          r: torch.Tensor,
                          v: torch.Tensor,
                          theta: torch.Tensor,
                          c: torch.Tensor,
                          target_image: torch.Tensor,
                          bmp_image: torch.Tensor,
                          opt_conf: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Override the optimization process to use separate optimizers for shape and appearance parameters.
        Shape parameters (x, y, r, theta) are optimized using shape loss.
        Appearance parameters (c, v) are optimized using color loss.
        
        Args:
            x, y, r, v, theta, c: Initial parameters
            target_image: Target image to match
            bmp_image: Base bitmap image for rasterization
            opt_conf: Optimization configuration
            
        Returns:
            Tuple of optimized parameters (x, y, r, v, theta, c)
        """
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 300)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Create separate optimizers for shape and appearance parameters
        shape_optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0)},
        ])
        
        appearance_optimizer = torch.optim.Adam([
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0)},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        print(f"Starting optimization for {num_iterations} iterations...")
        for epoch in range(num_iterations):
            shape_optimizer.zero_grad()
            appearance_optimizer.zero_grad()
            
            # Generate masks using shape parameters (x, y, r, theta)
            cached_masks = self._batched_soft_rasterize(
                bmp_image, x, y, r, theta,
                sigma=opt_conf.get("blur_sigma", 0.0)
            )
            
            # Render image using appearance parameters (c, v)
            rendered = self.render(cached_masks, v, c)
            
            # Compute losses
            total_loss, shape_loss, color_loss = self.compute_loss(
                rendered, target_image, cached_masks,
                x, y, r, v, theta, c
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            shape_optimizer.step()
            appearance_optimizer.step()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                      f"Shape Loss: {shape_loss.item():.4f}, "
                      f"Color Loss: {color_loss.item():.4f}")
        
        return x, y, r, v, theta, c 