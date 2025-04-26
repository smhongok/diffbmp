import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from core.renderer.vector_renderer import VectorRenderer
from typing import Tuple, Dict, Any
from torchvision import models
import piq

class MixRenderer(VectorRenderer):
    def __init__(self, canvas_size: Tuple[int, int], alpha_upper_bound: float = 0.5, device: torch.device = None, classify_svg: str = None):
        super().__init__(canvas_size, alpha_upper_bound, device)
        self.classify_svg = classify_svg
        vgg = models.vgg16(pretrained=True).features.eval().to(self.device)
        self.perc_layers = [3, 8]  # conv1_2, conv2_2
        self.vgg = vgg
        # ImageNet mean/std: [C,H,W] 후에 (1,3,1,1)로 브로드캐스트
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # 속성으로 저장하고 device로 이동
        self.vgg_mean = mean.view(1,3,1,1).to(self.device)
        self.vgg_std  = std .view(1,3,1,1).to(self.device)
        self.ssim = piq.SSIMLoss(kernel_size=11, reduction='mean').to(self.device)
        # Initialize LPIPS model for perceptual loss
        self.lpips = piq.LPIPS(reduction='mean').to(self.device)
        
        # 1차 미분용 Sobel 커널 정의
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]],
                          dtype=torch.float32, device=self.device)
        ky = torch.tensor([[1,  2,  1],
                           [0,  0,  0],
                           [-1, -2, -1]],
                          dtype=torch.float32, device=self.device)
        self.kx = kx.view(1,1,3,3)
        self.ky = ky.view(1,1,3,3)

    """
    Renderer using a combination of shape alignment loss (Mask IoU/Dice) for geometric parameters
    and masked L1 loss for color/alpha parameters.
    """
    def compute_shape_alignment_loss(self, 
                                   mask_pred: torch.Tensor, 
                                   mask_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target masks.
        
        Args:
            pred_masks: Predicted binary masks (B, H, W)
            target_masks: Target binary mask (H, W)
            
        Returns:
            IoU loss value (ℓmask = 1 - IoU)
        """
        # Expand target mask to match batch dimension
        target_masks = mask_gt.unsqueeze(0).expand_as(mask_pred)
        
        # Compute intersection and union
        mul_masks = mask_pred * target_masks
        intersection = torch.sum(mul_masks, dim=(1, 2))
        union = torch.sum(mask_pred + target_masks - mul_masks, dim=(1, 2)).float() + 1e-8
        
        # Compute IoU
        iou = intersection / union
        
        # Return mean IoU loss
        return 1.0 - torch.mean(iou)

    def compute_color_alpha_loss(self, 
                               rendered: torch.Tensor,
                               target: torch.Tensor,
                               mask_gt: torch.Tensor) -> torch.Tensor:
        # Get target mask (Mgt)
        m = mask_gt.unsqueeze(-1)
        return ( (rendered - target).abs() * m ).sum() / (m.sum() + 1e-8)
    
    def compute_edge_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Sobel edge L1 loss on grayscale renders.
        """
        # to shape (1,1,H,W)
        def to_gray(x):
            # x: (H,W,3) in [0,1]
            gray = 0.2989*x[...,0] + 0.5870*x[...,1] + 0.1140*x[...,2]
            return gray.unsqueeze(0).unsqueeze(0)
        r_gray = to_gray(rendered)
        t_gray = to_gray(target)

        # define sobel kernels
        device = rendered.device
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device)
        kx = kx.view(1,1,3,3)
        ky = ky.view(1,1,3,3)

        # convolve (padding=1 to keep size)
        grad_r_x = F.conv2d(r_gray, kx, padding=1)
        grad_r_y = F.conv2d(r_gray, ky, padding=1)
        grad_t_x = F.conv2d(t_gray, kx, padding=1)
        grad_t_y = F.conv2d(t_gray, ky, padding=1)

        # L1 difference
        loss_x = torch.abs(grad_r_x - grad_t_x).mean()
        loss_y = torch.abs(grad_r_y - grad_t_y).mean()
        return loss_x + loss_y
    
    def compute_curvature_loss(self, mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        # Approximate curvature via second derivatives on GT mask
        # ℓ_curv = L1(|∇²_pred| - |∇²_gt|)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                           dtype=torch.float32,device=self.device).view(1,1,3,3)
        p = mask_pred.unsqueeze(0).unsqueeze(0)
        g = mask_gt.unsqueeze(0).unsqueeze(0)
        lap_p = F.conv2d(p, lap, padding=1)
        lap_g = F.conv2d(g, lap, padding=1)
        return (lap_p.abs() - lap_g.abs()).abs().mean()
    
    def compute_perceptual(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        VGG feature MSE loss. 
        rendered, target: (H, W, 3), values in [0,1]
        """
        # (1,3,H,W) 로 포맷
        r = rendered.permute(2,0,1).unsqueeze(0)
        t = target  .permute(2,0,1).unsqueeze(0)
        # ImageNet 정규화
        r = (r - self.vgg_mean) / self.vgg_std
        t = (t - self.vgg_mean) / self.vgg_std

        loss = 0.0
        x, y = r, t
        # 한 번만 forward 하며, 지정된 레이어마다 MSE 계산
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if idx in self.perc_layers:
                loss = loss + F.mse_loss(x, y)
        return loss
    
    def compute_ssim_loss(self, rendered, target):
        # 입력: (H,W,3) → (1,3,H,W)
        r = rendered.permute(2,0,1).unsqueeze(0)
        t = target  .permute(2,0,1).unsqueeze(0)
        return 1 - self.ssim(r, t)
    
    def compute_boundary_loss(self, mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            
        Returns:
            Boundary loss value
        """
        if mask_pred.dim() == 5:
            # e.g. (B,1,H,W,3) → average over last dim
            mask_pred = mask_pred.mean(-1)            # now (B,1,H,W)
            mask_gt   = mask_gt.unsqueeze(0).mean(-1)  # maybe (1,1,H,W)
        elif mask_pred.dim() == 3:
            # (H,W,C) → (H,W)
            mask_pred = mask_pred.mean(-1)
            mask_gt   = mask_gt.mean(-1)
        # now handle 2D or 4D as above...
        if mask_pred.dim() == 2:
            m_pred = mask_pred.unsqueeze(0).unsqueeze(0)
            m_gt   = mask_gt.unsqueeze(0).unsqueeze(0)
        else:  # assume mask_pred.dim()==4
            m_pred = mask_pred
            m_gt   = mask_gt
            
         # 2) Sobel convolution
        grad_px = F.conv2d(m_pred, self.kx, padding=1)
        grad_py = F.conv2d(m_pred, self.ky, padding=1)
        grad_gx = F.conv2d(m_gt,   self.kx, padding=1)
        grad_gy = F.conv2d(m_gt,   self.ky, padding=1)

        # 3) L1 차이
        loss = (grad_px - grad_gx).abs().mean() + (grad_py - grad_gy).abs().mean()
        return loss
    
    def compute_perceptual_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using LPIPS.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            
        Returns:
            Perceptual loss value
        """
        # Convert to NCHW format for LPIPS
        rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        target_nchw = target.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Compute LPIPS loss
        perceptual_loss = self.lpips(rendered_nchw, target_nchw)
        
        return perceptual_loss
            
    def compute_boundary_loss(self, mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            
        Returns:
            Boundary loss value
        """
        if mask_pred.dim() == 5:
            # e.g. (B,1,H,W,3) → average over last dim
            mask_pred = mask_pred.mean(-1)            # now (B,1,H,W)
            mask_gt   = mask_gt.unsqueeze(0).mean(-1)  # maybe (1,1,H,W)
        elif mask_pred.dim() == 3:
            # (H,W,C) → (H,W)
            mask_pred = mask_pred.mean(-1)
            mask_gt   = mask_gt.mean(-1)
        # now handle 2D or 4D as above...
        if mask_pred.dim() == 2:
            m_pred = mask_pred.unsqueeze(0).unsqueeze(0)
            m_gt   = mask_gt.unsqueeze(0).unsqueeze(0)
        else:  # assume mask_pred.dim()==4
            m_pred = mask_pred
            m_gt   = mask_gt
            
         # 2) Sobel convolution
        grad_px = F.conv2d(m_pred, self.kx, padding=1)
        grad_py = F.conv2d(m_pred, self.ky, padding=1)
        grad_gx = F.conv2d(m_gt,   self.kx, padding=1)
        grad_gy = F.conv2d(m_gt,   self.ky, padding=1)

        # 3) L1 차이
        loss = (grad_px - grad_gx).abs().mean() + (grad_py - grad_gy).abs().mean()
        return loss
    
    def compute_loss(self, 
                    rendered: torch.Tensor, 
                    target: torch.Tensor, 
                    cached_masks: torch.Tensor,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    r: torch.Tensor,
                    v: torch.Tensor,
                    theta: torch.Tensor,
                    c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss for optimization.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            cached_masks: Generated masks (B, H, W)
            x, y, r, v, theta, c: Current parameter values
            
        Returns:
            Tuple of (total_loss, edge_loss, perceptual_loss, color_loss, boundary_loss)
        """
        
        gt_mask   = (target[...,0] > 0).float()    # 2D: H×W

        # Compute individual losses
        edge_loss = self.compute_edge_loss(rendered, target)
        boundary_loss = self.compute_boundary_loss(rendered, target)
        perceptual_loss = self.compute_perceptual_loss(rendered, target)
        color_loss = self.compute_color_alpha_loss(rendered, target, gt_mask)
        
        # Adjust weights based on SVG classification
        if self.classify_svg == 'curve':
            # For curves, emphasize shape and edge loss
            w_edge = 0.5
            w_boundary = 0.1
            w_perceptual = 0.2
            w_color = 0.1
        elif self.classify_svg == 'text':
            # For text, emphasize edge and perceptual loss
            w_edge = 0.2
            w_boundary = 0.3
            w_perceptual = 0.2
            w_color = 0.1
        else:
            # Default weights
            w_edge = 0.3
            w_boundary = 0.1
            w_perceptual = 0.2
            w_color = 0.2
        
        # Compute total loss
        total_loss = (w_edge * edge_loss + 
                     w_perceptual * perceptual_loss + 
                     w_color * color_loss +
                     w_boundary * boundary_loss)
        
        return total_loss, edge_loss, boundary_loss, perceptual_loss, color_loss
    
    def compute_shape_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(rendered, target)
    
    def compute_appearance_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute appearance loss between rendered and target images.
        
        Args:
            rendered: Rendered image tensor (H, W, 3)
            target: Target image tensor (H, W, 3)
            
        Returns:
            Appearance loss value
        """
        gt_mask   = (target[...,0] > 0).float()    # 2D: H×W
        color_loss = self.compute_color_alpha_loss(rendered, target, gt_mask)
        
        total_loss = color_loss
        
        return total_loss, color_loss

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
        for epoch in tqdm(range(num_iterations)):
            # Step 1: Optimize shape parameters
            shape_optimizer.zero_grad()
            
            # Generate masks using shape parameters (x, y, r, theta)
            cached_masks = self._batched_soft_rasterize(
                bmp_image, x, y, r, theta,
                sigma=opt_conf.get("blur_sigma", 0.0)
            )
            
            # Render image using appearance parameters (v, c)
            rendered = self.render(cached_masks, v, c)
            #rendered = self.render(cached_masks, v.detach(), c.detach())
            
            # Compute shape loss
            shape_loss = self.compute_shape_loss(
                rendered, target_image#, cached_masks
            )
            
            # Backward pass for shape parameters
            shape_loss.backward(retain_graph=True)
            shape_optimizer.step()
            
            # Step 2: Optimize appearance parameters
            appearance_optimizer.zero_grad()
            
            # Re-render with updated shape parameters
            cached_masks = self._batched_soft_rasterize(
                bmp_image, x.detach(), y.detach(), r.detach(), theta.detach(),
                sigma=opt_conf.get("blur_sigma", 0.0)
            )
            rendered = self.render(cached_masks, v, c)
            
            # Compute appearance loss
            appearance_loss, color_loss = self.compute_appearance_loss(
                rendered, target_image
            )
            
            # Backward pass for appearance parameters
            appearance_loss.backward()
            appearance_optimizer.step()
            
            # Clamp parameters
            with torch.no_grad():
                x.clamp_(0, self.W)
                y.clamp_(0, self.H)
                r.clamp_(2, min(self.H, self.W) // 4)
                theta.clamp_(0, 2 * np.pi)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Shape Loss: {shape_loss.item():.4f}, "
                      f"Appearance Loss: {appearance_loss.item():.4f}")
        
        return x, y, r, v, theta, c 