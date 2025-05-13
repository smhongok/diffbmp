import torch
import torch.nn.functional as F
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
import numpy as np
import math
import os
import datetime
from PIL import Image

class FreqRenderer(VectorRenderer):
    """
    Renderer using MSE loss for optimization.
    This is the same as the base VectorRenderer implementation.
    """
    def __init__(self, canvas_size, S, alpha_upper_bound=0.5, device='cuda', use_fp16=True, gamma=1.0, output_path=None):
        super().__init__(canvas_size, S, alpha_upper_bound, device, use_fp16, gamma, output_path)
        
    def compute_shape_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(rendered, target)
        
    def focal_frequency_loss(self, rendered: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        """
        Focal Frequency Loss (Jiang et al. ICCV 2021).
        - rendered, target: (H, W, 3) or (3, H, W), values in [0,1]
        """
        # ensure shape (B=1), C, H, W
        # here assume no batch, so add batch dim & permute to (1, C, H, W)
        x = rendered
        y = target
        if x.dim() == 3:  # (H, W, C) → (1, C, H, W)
            x = x.permute(2, 0, 1).unsqueeze(0)
            y = y.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        
        pad = max(x.shape[2], x.shape[3]) // 2
        x_pad = F.pad(x, (pad,pad,pad,pad), mode='reflect')
        y_pad = F.pad(y, (pad,pad,pad,pad), mode='reflect')

        # real-to-complex FFT2
        Xf = torch.fft.rfft(x_pad, 2, norm='ortho')
        Yf = torch.fft.rfft(y_pad, 2, norm='ortho')
        mag = torch.abs(Xf - Yf)

        # focal weights: emphasize large errors
        weight = mag ** gamma
        weight[..., 0, 0] = 0.0  # DC 성분 가중치 강제 0
        weight = weight / (weight.sum(dim=(-2,-1), keepdim=True) + eps)

        # FFL: weighted sum of squared magnitudes
        loss = (weight * (mag ** 2)).sum()
        return loss

        
    def compute_frequency_loss(self, rendered: torch.Tensor, target: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        Compute MSE loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            MSE loss value
        """
        # Ensure tensors are in consistent precision
        if self.use_fp16:
            # If target is in FP32, convert rendered to FP32
            if target.dtype == torch.float32:
                rendered = rendered.float()
            # If rendered is in FP16, convert target to FP16
            elif rendered.dtype == torch.float16 and target.dtype != torch.float16:
                target = target.half()
        else:
            # In FP32 mode, ensure everything is float32
            rendered = rendered.float()
            target = target.float()
        
        return self.focal_frequency_loss(rendered, target, gamma=gamma)

    def optimize_parameters(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          r: torch.Tensor,
                          v: torch.Tensor,
                          theta: torch.Tensor,
                          c: torch.Tensor,
                          target_image: torch.Tensor,
                          opt_conf: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Override the optimization process to use separate optimizers for shape and appearance parameters.
        Shape parameters (x, y, r, theta) are optimized using shape loss.
        Appearance parameters (c, v) are optimized using color loss.
        
        Args:
            x, y, r, v, theta, c: Initial parameters
            target_image: Target image to match
            opt_conf: Optimization configuration
            
        Returns:
            Tuple of optimized parameters (x, y, r, v, theta, c)
        """
        # Get optimization parameters from config
        num_iterations = opt_conf.get("num_iterations", 100)
        lr_conf = opt_conf["learning_rate"]
        lr = lr_conf.get("default", 0.1)
        
        # Create output directory for saving images if it doesn't exist
        save_image_intervals = [1, 5, 10, 20, 50, 100]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create separate optimizers for shape and appearance parameters
        shape_optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0) * (1000.0/x.numel())},
        ])
        
        appearance_optimizer = torch.optim.Adam([
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        # Only use mixed precision in FP16 mode
        shape_scaler = GradScaler('cuda') if self.use_fp16 else None
        appearance_scaler = GradScaler('cuda') if self.use_fp16 else None
        
        T = 2.0  # 온도 하이퍼파라미터
        print(f"Starting optimization for {num_iterations} iterations...")
        L1 = [None, None]
        L2 = [None, None]
        for epoch in tqdm(range(num_iterations)):
            # Define context manager based on precision mode
            shape_context = autocast('cuda') if self.use_fp16 else nullcontext()
            appearance_context = autocast('cuda') if self.use_fp16 else nullcontext()
            
            # Step 1: Optimize shape parameters
            shape_optimizer.zero_grad()
            with shape_context:
                if opt_conf.get("multi_level", False):
                    rendered = self.render(self.S, v, c)
                else:
                    # Generate masks using shape parameters (x, y, r, theta)
                    cached_masks = self._batched_soft_rasterize(
                        x, y, r, theta,
                        sigma=opt_conf.get("blur_sigma", 0.0)
                    )
                    
                    # Render image using appearance parameters (v, c)
                    rendered = self.render(cached_masks, v, c)
                    
                    # Save image at specified epochs after first optimizer
                    if opt_conf.get("save_epoch", False) and epoch + 1 in save_image_intervals:
                        output_path = os.path.join(self.output_path, f"freq_epoch_{epoch+1}_{timestamp}.jpg")
                        rendered_np = rendered.detach().cpu().numpy()
                        rendered_np = (rendered_np * 255).astype(np.uint8)
                        Image.fromarray(rendered_np).save(output_path)
                        del rendered_np
                        
                # Compute shape loss
                shape_loss = self.compute_shape_loss(
                    rendered, target_image
                )
                
                # Backward pass for shape parameters - handled differently based on precision mode
                if self.use_fp16:
                    shape_scaler.scale(shape_loss).backward(retain_graph=True)
                    shape_scaler.step(shape_optimizer)
                    shape_scaler.update()
                    del rendered, cached_masks
                    torch.cuda.empty_cache()
                else:
                    shape_loss.backward(retain_graph=True)
                    shape_optimizer.step()
            
            # Step 2: Optimize appearance parameters
            appearance_optimizer.zero_grad()
            
            with appearance_context:
                if opt_conf.get("multi_level", False):
                    rendered = self.render(self.S, v, c)
                else:
                    # Re-render with updated shape parameters
                    cached_masks = self._batched_soft_rasterize(
                        x.detach(), y.detach(), r.detach(), theta.detach(),
                        sigma=opt_conf.get("blur_sigma", 0.0)
                    )
                    rendered = self.render(cached_masks, v, c)
                
                # Compute appearance loss
                mse_loss = self.compute_shape_loss(
                    rendered, target_image
                )
                frequency_loss = self.compute_frequency_loss(
                    rendered, target_image, gamma=self.gamma
                )
                L1[epoch % 2] = mse_loss
                L2[epoch % 2] = frequency_loss
                if opt_conf.get("do_dwa", True):
                    # 지난 two epochs의 loss 변화율을 비교해 자동 weight 산출
                    if epoch > 2:
                        w1 = math.exp((L1[-1]/L1[-2]) / T)
                        w2 = math.exp((L2[-1]/L2[-2]) / T)
                        norm = w1 + w2
                        alpha = 2 * w1 / norm
                        beta  = 2 * w2 / norm
                        appearance_loss = alpha * mse_loss + beta * frequency_loss
                    else:
                        appearance_loss = mse_loss 
                else:
                    appearance_loss = mse_loss + frequency_loss
                
                # Backward pass for appearance parameters
                if self.use_fp16:
                    appearance_scaler.scale(appearance_loss).backward()
                    appearance_scaler.step(appearance_optimizer)
                    appearance_scaler.update()
                    del rendered, cached_masks
                    torch.cuda.empty_cache()
                else:
                    appearance_loss.backward()
                    appearance_optimizer.step()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Shape Loss: {shape_loss.item():.4f}, Appearance Loss: {appearance_loss.item():.4f}")
            
        return x, y, r, v, theta, c 
        
        '''
        # Create separate optimizers for shape and appearance parameters
        optimizer = torch.optim.Adam([
            {'params': x, 'lr': lr*lr_conf.get("gain_x", 1.0)},
            {'params': y, 'lr': lr*lr_conf.get("gain_y", 1.0)},
            {'params': r, 'lr': lr*lr_conf.get("gain_r", 1.0)},
            {'params': theta, 'lr': lr*lr_conf.get("gain_theta", 1.0) * (1000.0/x.numel())},
            {'params': v, 'lr': lr*lr_conf.get("gain_v", 1.0) * (1000.0/x.numel())},
            {'params': c, 'lr': lr*lr_conf.get("gain_c", 1.0)},
        ])
        
        # Only use mixed precision in FP16 mode
        scaler = GradScaler('cuda') if self.use_fp16 else None
        
        T = 2.0  # 온도 하이퍼파라미터
        print(f"Starting optimization for {num_iterations} iterations...")
        L1 = []
        L2 = []
        mse_loss, frequency_loss = 0, 0
        for epoch in tqdm(range(num_iterations)):
            context = autocast('cuda') if self.use_fp16 else nullcontext()
            with context:
                # Step 1: Optimize shape parameters
                optimizer.zero_grad()
                if opt_conf.get("multi_level", False):
                    rendered = self.render(self.S, v, c)
                else:
                    # Generate masks using shape parameters (x, y, r, theta)
                    cached_masks = self._batched_soft_rasterize(
                        x, y, r, theta,
                        sigma=opt_conf.get("blur_sigma", 0.0)
                    )
                    
                    # Render image using appearance parameters (v, c)
                    rendered = self.render(cached_masks, v, c)
            
                # Compute shape loss
                mse_loss = F.mse_loss(rendered, target_image)
                frequency_loss = self.compute_frequency_loss(rendered, target_image, gamma=self.gamma) #1.2 + 0.8 * (epoch / num_iterations))
                L1.append(mse_loss)
                L2.append(frequency_loss)
                if opt_conf.get("do_dwa", True):
                    # 지난 two epochs의 loss 변화율을 비교해 자동 weight 산출
                    if epoch > 2:
                        w1 = math.exp((L1[-1]/L1[-2]) / T)
                        w2 = math.exp((L2[-1]/L2[-2]) / T)
                        norm = w1 + w2
                        alpha = 2 * w1 / norm
                        beta  = 2 * w2 / norm
                        loss = alpha * mse_loss + beta * frequency_loss
                    else:
                        loss = mse_loss 
                else:
                    loss = mse_loss + frequency_loss
                    
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, MSE Loss: {mse_loss:.4f}, Frequency Loss: {frequency_loss:.4f}")
                
                # Backward pass for shape parameters - handled differently based on precision mode
                if self.use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Clean up memory if using FP16
                if self.use_fp16:
                    del rendered, cached_masks
                    torch.cuda.empty_cache()
                
                # Clamp parameters
                with torch.no_grad():
                    x.clamp_(0, self.W)
                    y.clamp_(0, self.H)
                    r.clamp_(2, min(self.H, self.W) // 4)
                    theta.clamp_(0, 2 * np.pi)
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return x, y, r, v, theta, c 
        '''