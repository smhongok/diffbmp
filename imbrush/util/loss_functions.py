"""
Flexible loss function registry for image-as-brush rendering optimization.
Supports various loss types and weighted combinations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable
import torchvision.models as models


class LossRegistry:
    """
    Registry for various loss functions used in rendering optimization.
    Each loss function has a consistent signature for easy composition.
    """
    
    @staticmethod
    def mse_loss(rendered: torch.Tensor, 
                 target: torch.Tensor, 
                 mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Mean Squared Error (L2) loss.
        
        Args:
            rendered: Rendered image (H, W, C)
            target: Target image (H, W, C)
            mask: Optional binary mask (H, W)
        """
        if mask is not None:
            rendered_masked = rendered[mask]
            target_masked = target[mask]
            if rendered_masked.numel() > 0:
                return F.mse_loss(rendered_masked, target_masked)
            else:
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        return F.mse_loss(rendered, target)
    
    @staticmethod
    def l1_loss(rendered: torch.Tensor, 
                target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Mean Absolute Error (L1) loss.
        More robust to outliers than MSE.
        """
        if mask is not None:
            rendered_masked = rendered[mask]
            target_masked = target[mask]
            if rendered_masked.numel() > 0:
                return F.l1_loss(rendered_masked, target_masked)
            else:
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        return F.l1_loss(rendered, target)
    
    @staticmethod
    def huber_loss(rendered: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None,
                   delta: float = 1.0,
                   **kwargs) -> torch.Tensor:
        """
        Huber loss (smooth L1).
        Combines benefits of L1 and L2: quadratic for small errors, linear for large.
        """
        if mask is not None:
            rendered_masked = rendered[mask]
            target_masked = target[mask]
            if rendered_masked.numel() > 0:
                return F.smooth_l1_loss(rendered_masked, target_masked, beta=delta)
            else:
                return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        return F.smooth_l1_loss(rendered, target, beta=delta)
    
    @staticmethod
    def perceptual_loss(rendered: torch.Tensor,
                       target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       vgg_model: Optional[torch.nn.Module] = None,
                       **kwargs) -> torch.Tensor:
        """
        VGG-based perceptual loss.
        Measures high-level feature similarity rather than pixel-wise.
        
        Note: Requires images in (B, C, H, W) format with values in [0, 1]
        """
        if vgg_model is None:
            # Lazy initialization - will be cached in compute_loss
            return torch.tensor(0.0, device=rendered.device, requires_grad=True)
        
        # Convert (H, W, C) -> (1, C, H, W)
        rendered_input = rendered.permute(2, 0, 1).unsqueeze(0)
        target_input = target.permute(2, 0, 1).unsqueeze(0)
        
        # Clamp to [0, 1] range for stability
        rendered_input = torch.clamp(rendered_input, 0.0, 1.0)
        target_input = torch.clamp(target_input, 0.0, 1.0)
        
        # VGG normalization (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=rendered.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rendered.device).view(1, 3, 1, 1)
        
        rendered_normalized = (rendered_input - mean) / std
        target_normalized = (target_input - mean) / std
        
        # Extract features (target should not require gradients)
        with torch.no_grad():
            target_features = vgg_model(target_normalized)
        
        rendered_features = vgg_model(rendered_normalized)
        
        # Compute MSE on features and normalize by feature dimension
        # This makes the loss scale more comparable to pixel-space MSE
        loss = F.mse_loss(rendered_features, target_features.detach())
        
        # Scale down to make it comparable to pixel MSE (typical VGG features are ~100x larger)
        loss = loss / 100.0
        
        return loss
    
    @staticmethod
    def ssim_loss(rendered: torch.Tensor,
                  target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  window_size: int = 11,
                  **kwargs) -> torch.Tensor:
        """
        Structural Similarity Index (SSIM) loss.
        Measures perceptual similarity based on luminance, contrast, and structure.
        
        WARNING: This loss function is currently DISABLED due to optimization issues.
        SSIM loss causes the total loss to INCREASE during training instead of decreasing.
        
        Issue: SSIM focuses on local structural similarity which conflicts with
        pixel-level optimization. The loss diverges rather than converges.
        Test results showed 21.7% increase over 100 iterations.
        """
        raise NotImplementedError(
            "SSIM loss is currently disabled due to optimization divergence. "
            "The loss increases during training instead of decreasing. "
            "Please use MSE, L1, Huber, or perceptual loss instead."
        )
    
    @staticmethod
    def edge_loss(rendered: torch.Tensor,
                  target: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  **kwargs) -> torch.Tensor:
        """
        Edge-based loss using Sobel filters.
        Emphasizes sharp boundaries and edges.
        
        WARNING: This loss function is currently DISABLED due to optimization issues.
        The edge loss gradient conflicts with MSE/perceptual losses, causing
        the total loss to increase during training instead of decreasing.
        
        Issue: Edge loss focuses on high-frequency details which interferes with
        learning the overall structure. Needs redesign before use.
        """
        raise NotImplementedError(
            "Edge loss is currently disabled due to gradient conflicts with other losses. "
            "The loss increases during optimization instead of decreasing. "
            "Please use MSE, perceptual, or other loss functions instead."
        )
    
    @staticmethod
    def alpha_loss(rendered_alpha: torch.Tensor,
                   target_alpha: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   **kwargs) -> torch.Tensor:
        """
        Alpha channel loss for transparency.
        Used when target has alpha channel.
        """
        # Handle shape: (H, W) or (H, W, 1)
        if rendered_alpha.dim() == 3 and rendered_alpha.shape[2] == 1:
            rendered_alpha = rendered_alpha.squeeze(-1)
        if target_alpha.dim() == 3 and target_alpha.shape[2] == 1:
            target_alpha = target_alpha.squeeze(-1)
        
        return F.mse_loss(rendered_alpha, target_alpha)


class LossComposer:
    """
    Composes multiple loss functions with configurable weights.
    """
    
    def __init__(self, loss_config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize loss composer from config.
        
        Args:
            loss_config: Dictionary with loss configuration
                Example:
                {
                    "type": "combined",
                    "components": [
                        {"name": "mse", "weight": 0.7},
                        {"name": "edge", "weight": 0.3}
                    ]
                }
        """
        self.device = device
        self.loss_config = loss_config
        self.loss_type = loss_config.get("type", "mse")
        
        # Initialize VGG model for perceptual loss if needed
        self.vgg_model = None
        if self.loss_type == "combined":
            components = loss_config.get("components", [])
            if any(c.get("name") == "perceptual" for c in components):
                self._init_vgg_model()
        elif self.loss_type == "perceptual":
            self._init_vgg_model()
    
    def _init_vgg_model(self):
        """Initialize VGG model for perceptual loss."""
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_model = vgg.to(self.device)
    
    def compute_loss(self,
                    rendered: torch.Tensor,
                    target: torch.Tensor,
                    rendered_alpha: Optional[torch.Tensor] = None,
                    target_alpha: Optional[torch.Tensor] = None,
                    mask: Optional[torch.Tensor] = None,
                    return_components: bool = False,
                    **kwargs):
        """
        Compute loss based on configuration.
        
        Args:
            rendered: Rendered RGB image (H, W, 3)
            target: Target RGB image (H, W, 3)
            rendered_alpha: Optional rendered alpha channel (H, W)
            target_alpha: Optional target alpha channel (H, W)
            mask: Optional mask for valid pixels (H, W)
            return_components: If True, return (total_loss, components_dict)
            
        Returns:
            Total loss value, or (total_loss, components_dict) if return_components=True
        """
        loss_type = self.loss_config.get("type", "mse")
        
        if loss_type == "combined":
            return self._compute_combined_loss(rendered, target, rendered_alpha, 
                                               target_alpha, mask, return_components, **kwargs)
        else:
            # Single loss function
            total_loss = self._compute_single_loss(loss_type, rendered, target, 
                                                   rendered_alpha, target_alpha, mask, **kwargs)
            if return_components:
                return total_loss, {loss_type: total_loss.item()}
            return total_loss
    
    def _compute_single_loss(self,
                            loss_name: str,
                            rendered: torch.Tensor,
                            target: torch.Tensor,
                            rendered_alpha: Optional[torch.Tensor],
                            target_alpha: Optional[torch.Tensor],
                            mask: Optional[torch.Tensor],
                            **kwargs) -> torch.Tensor:
        """Compute a single loss function."""
        loss_fn = getattr(LossRegistry, f"{loss_name}_loss", None)
        if loss_fn is None:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        # Add VGG model if needed
        if loss_name == "perceptual":
            kwargs['vgg_model'] = self.vgg_model
        
        return loss_fn(rendered, target, mask=mask, **kwargs)
    
    def _compute_combined_loss(self,
                              rendered: torch.Tensor,
                              target: torch.Tensor,
                              rendered_alpha: Optional[torch.Tensor],
                              target_alpha: Optional[torch.Tensor],
                              mask: Optional[torch.Tensor],
                              return_components: bool = False,
                              **kwargs):
        """Compute weighted combination of multiple losses."""
        components = self.loss_config.get("components", [])
        
        if not components:
            # Fallback to MSE if no components specified
            total_loss = LossRegistry.mse_loss(rendered, target, mask=mask)
            if return_components:
                return total_loss, {"mse": total_loss.item()}
            return total_loss
        
        total_loss = None
        loss_components = {}
        
        for component in components:
            loss_name = component.get("name")
            weight = component.get("weight", 1.0)
            
            # Skip if weight is zero or very small
            if abs(weight) < 1e-8:
                if return_components:
                    loss_components[loss_name] = 0.0
                continue
            
            # Special handling for alpha loss
            if loss_name == "alpha":
                if rendered_alpha is not None and target_alpha is not None:
                    loss_value = LossRegistry.alpha_loss(rendered_alpha, target_alpha, mask=mask)
                    weighted_loss = weight * loss_value
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                    if return_components:
                        loss_components[loss_name] = loss_value.item()
            else:
                # RGB losses
                loss_fn = getattr(LossRegistry, f"{loss_name}_loss", None)
                if loss_fn is None:
                    print(f"Warning: Unknown loss function '{loss_name}', skipping")
                    continue
                
                # Add VGG model if needed
                loss_kwargs = kwargs.copy()
                if loss_name == "perceptual":
                    loss_kwargs['vgg_model'] = self.vgg_model
                
                loss_value = loss_fn(rendered, target, mask=mask, **loss_kwargs)
                weighted_loss = weight * loss_value
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                if return_components:
                    loss_components[loss_name] = loss_value.item()
        
        # If no valid loss computed, return zero
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if return_components:
            return total_loss, loss_components
        return total_loss
