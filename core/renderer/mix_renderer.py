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
        Compute Dice loss between predicted and target masks.
        
        Args:
            pred_masks: Predicted binary masks (B, H, W)
            target_masks: Target binary mask (H, W)
            
        Returns:
            Dice loss value
        """
        # Expand target mask to match batch dimension
        target_masks = target_masks.unsqueeze(0).expand_as(pred_masks)
        
        # Compute intersection and union
        intersection = torch.sum(pred_masks * target_masks, dim=(1, 2))
        pred_sum = torch.sum(pred_masks, dim=(1, 2))
        target_sum = torch.sum(target_masks, dim=(1, 2))
        
        # Compute Dice coefficient for each mask
        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
        
        # Return mean Dice loss
        return 1.0 - torch.mean(dice)

    def compute_color_alpha_loss(self, 
                               rendered: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
        """
        Compute masked L1 loss for color and alpha values.
        Only compute loss in regions where the mask is active.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            masks: Binary masks (B, H, W)
            
        Returns:
            Masked L1 loss value
        """
        # Create mask for valid regions (where mask value > threshold)
        valid_mask = (masks > 0.1).float()
        
        # Compute L1 loss only in valid regions
        diff = torch.abs(rendered - target)
        masked_diff = diff * valid_mask.unsqueeze(-1)
        
        # Normalize by number of valid pixels
        num_valid = torch.sum(valid_mask) + 1e-8
        return torch.sum(masked_diff) / num_valid

    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    cached_masks: torch.Tensor,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss using shape alignment for geometric parameters and
        masked L1 loss for color/alpha parameters.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            Combined loss value
        """
        # Extract target mask from target image (assuming first channel represents mask)
        target_mask = target[..., 0]
        
        # Compute shape alignment loss
        shape_loss = self.compute_shape_alignment_loss(cached_masks, target_mask)
        
        # Compute color/alpha loss
        color_loss = self.compute_color_alpha_loss(rendered, target, cached_masks)
        
        # Combine losses with weighting
        shape_weight = 0.7
        color_weight = 0.3
        
        return shape_weight * shape_loss + color_weight * color_loss

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
            
            # Generate masks
            cached_masks = self._batched_soft_rasterize(
                bmp_image, x, y, r, theta,
                sigma=opt_conf.get("blur_sigma", 0.0)
            )
            
            # Render image
            rendered = self.render(cached_masks, v, c)
            
            # Compute loss
            loss = self.compute_loss(rendered, target_image, cached_masks,
                                   x, y, r, v, theta, c)
            loss.backward()
            
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
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return x, y, r, v, theta, c 