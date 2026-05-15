"""Compatibility entrypoint for the real CUDA bitmap primitive rasterizer.

AutoResearchClaw-generated experiment code may look for a compact module named
``diffbmp_cuda``.  The actual implementation in this checkout lives in
``cuda_tile_rasterizer.TileRasterizer``.  This module intentionally provides only
thin wrappers around that CUDA extension; it does not implement a Python or
PyTorch renderer.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import torch

from cuda_tile_rasterizer import TileRasterizer


CUDA = True
IS_CUDA_EXTENSION = True
SUPPORTS_CUDA = True
DEFORMABLE_CUDA_KERNEL = True
SUPPORTS_DEFORMABLE_CUDA = True
supports_deformable_cuda = True
__version__ = "local-cuda-dct-deformable-adapter"


def _first_tensor(*values: torch.Tensor | None) -> torch.Tensor | None:
    for value in values:
        if value is not None:
            return value
    return None


def _default_bitmaps(num_primitives: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """Create a smooth opaque bitmap template when callers pass geometry only."""
    coords = torch.linspace(-1.0, 1.0, patch_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx * xx + yy * yy)
    alpha = torch.sigmoid((0.82 - radius) * 18.0)
    return alpha.expand(num_primitives, patch_size, patch_size).contiguous()


def _as_state(primitive_state: dict[str, torch.Tensor] | None, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
    if primitive_state is not None:
        centers = _first_tensor(primitive_state.get("centers"), primitive_state.get("center"))
        scales = _first_tensor(primitive_state.get("scales"), primitive_state.get("scale"))
        rotations = _first_tensor(
            primitive_state.get("rotations"),
            primitive_state.get("rotation"),
            primitive_state.get("angles"),
            primitive_state.get("angle"),
        )
        colors = _first_tensor(primitive_state.get("colors"), primitive_state.get("color"))
        opacities = _first_tensor(primitive_state.get("opacities"), primitive_state.get("opacity"))
        bitmaps = _first_tensor(primitive_state.get("bitmaps"), primitive_state.get("bitmap"), primitive_state.get("templates"))
        state = {
            "centers": centers,
            "scales": scales,
            "rotations": rotations,
            "colors": colors,
            "opacities": opacities,
            "bitmaps": bitmaps,
        }
        missing = [key for key, value in state.items() if value is None and key != "bitmaps"]
        if missing:
            raise ValueError(f"Missing primitive state tensors: {missing}")
        if state["bitmaps"] is None:
            assert centers is not None
            state["bitmaps"] = _default_bitmaps(int(centers.shape[0]), 32, centers.device)
        return state  # type: ignore[return-value]
    state = {
        "centers": _first_tensor(kwargs.get("centers"), kwargs.get("center")),
        "scales": _first_tensor(kwargs.get("scales"), kwargs.get("scale")),
        "rotations": _first_tensor(kwargs.get("rotations"), kwargs.get("rotation"), kwargs.get("angles"), kwargs.get("angle")),
        "colors": _first_tensor(kwargs.get("colors"), kwargs.get("color")),
        "opacities": _first_tensor(kwargs.get("opacities"), kwargs.get("opacity")),
        "bitmaps": _first_tensor(kwargs.get("bitmaps"), kwargs.get("bitmap"), kwargs.get("templates")),
    }
    missing = [key for key, value in state.items() if value is None and key != "bitmaps"]
    if missing:
        raise ValueError(f"Missing primitive state tensors: {missing}")
    if state["bitmaps"] is None:
        centers = state["centers"]
        assert centers is not None
        state["bitmaps"] = _default_bitmaps(int(centers.shape[0]), 32, centers.device)
    return state  # type: ignore[return-value]


@lru_cache(maxsize=32)
def _rasterizer(height: int, width: int, tile_size: int, num_primitives: int) -> TileRasterizer:
    return TileRasterizer(
        height,
        width,
        tile_size=tile_size,
        sigma=0.0,
        alpha_upper_bound=1.0,
        max_prims_per_pixel=max(16, min(4096, num_primitives)),
        num_primitives=num_primitives,
        use_fp16=False,
    )


def _tile_mapping(height: int, width: int, tile_size: int, num_primitives: int, device: torch.device) -> torch.Tensor:
    tiles_y = max(1, math.ceil(height / tile_size))
    tiles_x = max(1, math.ceil(width / tile_size))
    num_tiles = tiles_y * tiles_x
    offsets = torch.arange(
        0,
        (num_tiles + 1) * num_primitives,
        num_primitives,
        device=device,
        dtype=torch.int32,
    )
    indices = torch.arange(num_primitives, device=device, dtype=torch.int32).repeat(num_tiles)
    return torch.cat(
        [
            torch.tensor([len(offsets), len(indices)], device=device, dtype=torch.int32),
            offsets,
            indices,
        ]
    )


def _field_to_dct_coeffs(field: torch.Tensor) -> torch.Tensor:
    """Project a dense [P,2,H,W] displacement field to the renderer's 8 DCT modes."""
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError(f"Expected deformation field [P,2,H,W], got {tuple(field.shape)}")
    p, _, h, w = field.shape
    device = field.device
    dtype = field.dtype
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / max(h, 1)
    x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / max(w, 1)
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
    coeffs = []
    # The CUDA kernel applies tanh(max_disp * raw); for small fields the
    # projection below is a stable low-frequency approximation and remains
    # differentiable back to the generated dense field.
    raw = torch.atanh(torch.clamp(field, -0.95, 0.95))
    for fu, fv in modes:
        bx = torch.cos(math.pi * fu * x)
        by = torch.cos(math.pi * fv * y)
        basis = by[:, None] * bx[None, :]
        denom = basis.square().mean().clamp_min(1e-6)
        coeff = (raw * basis.view(1, 1, h, w)).mean(dim=(-2, -1)) / denom
        coeffs.append(coeff)
    return torch.stack(coeffs, dim=1).reshape(p, 8, 2).contiguous()


def render_cuda(
    primitive_state: dict[str, torch.Tensor] | None = None,
    height: int | None = None,
    width: int | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    image_size: tuple[int, int] | None = None,
    deformation_grid: torch.Tensor | None = None,
    deformations: torch.Tensor | None = None,
    deformation_enabled: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Render bitmap primitives through the compiled CUDA TileRasterizer.

    Returns a tensor with shape ``[1, 4, H, W]`` for generated harnesses that
    expect channel-first RGBA output.
    """
    state = _as_state(primitive_state, kwargs)
    centers = state["centers"].float().contiguous()
    scales = state["scales"].float().contiguous()
    rotations = state["rotations"].float().reshape(-1).contiguous()
    colors = state["colors"].float().contiguous()
    opacities = state["opacities"].float().reshape(-1).contiguous()
    bitmaps = state["bitmaps"].float().contiguous()
    if bitmaps.ndim == 4 and bitmaps.shape[1] == 1:
        templates = bitmaps[:, 0]
    elif bitmaps.ndim == 3:
        templates = bitmaps
    else:
        raise ValueError(f"Unsupported bitmap tensor shape: {tuple(bitmaps.shape)}")

    if isinstance(height, (tuple, list)) and width is None:
        image_size = (int(height[0]), int(height[1]))
        height = None
    if image_size is None:
        canvas_hw = kwargs.get("canvas_hw")
        if isinstance(canvas_hw, (tuple, list)) and len(canvas_hw) == 2:
            image_size = (int(canvas_hw[0]), int(canvas_hw[1]))
    if image_size is not None:
        height = int(image_size[0])
        width = int(image_size[1])
    h = int(height if height is not None else image_height)
    w = int(width if width is not None else image_width)
    if h <= 0 or w <= 0:
        raise ValueError("height/width must be positive")
    if centers.device.type != "cuda":
        raise RuntimeError("render_cuda requires CUDA tensors")

    p = int(centers.shape[0])
    means = centers.clone()
    means[:, 0] = means[:, 0] * float(w)
    means[:, 1] = means[:, 1] * float(h)
    if scales.ndim == 2:
        radii = scales.mean(dim=1) * float(min(h, w))
    else:
        radii = scales.reshape(-1) * float(min(h, w))
    radii = radii.clamp_min(1.0).contiguous()

    k_h, k_w = int(templates.shape[-2]), int(templates.shape[-1])
    colors_orig = colors.view(p, 1, 1, 3).expand(p, k_h, k_w, 3).contiguous()
    global_bmp_sel = torch.arange(p, device=centers.device, dtype=torch.long)
    lr_conf = torch.tensor([0.1, 1, 1, 1, 1, 1, 1], device=centers.device, dtype=torch.float32)
    tile_size = 16
    mapping = _tile_mapping(h, w, tile_size, p, centers.device)

    field = deformation_grid if deformation_grid is not None else deformations
    if deformation_enabled and field is not None:
        deform_coeffs = _field_to_dct_coeffs(field.float().contiguous())
        deform_max_disp = 1.0
    else:
        deform_coeffs = None
        deform_max_disp = 0.0

    rasterizer = _rasterizer(h, w, tile_size, p)
    out_color, out_alpha = rasterizer(
        means,
        radii,
        rotations,
        opacities,
        colors,
        colors_orig,
        templates.contiguous(),
        global_bmp_sel,
        0.0,
        lr_conf,
        mapping,
        deform_coeffs=deform_coeffs,
        deform_max_disp=deform_max_disp,
    )
    if out_color.ndim == 3:
        color_chw = out_color.permute(2, 0, 1).unsqueeze(0)
    elif out_color.ndim == 4:
        color_chw = out_color.permute(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Unexpected CUDA color output shape: {tuple(out_color.shape)}")
    if out_alpha.ndim == 2:
        alpha = out_alpha.unsqueeze(0).unsqueeze(0)
    elif out_alpha.ndim == 3:
        alpha = out_alpha.unsqueeze(1)
    else:
        raise RuntimeError(f"Unexpected CUDA alpha output shape: {tuple(out_alpha.shape)}")
    return torch.cat([color_chw.float(), alpha.float()], dim=1).contiguous()


render = render_cuda
rasterize = render_cuda
rasterize_cuda = render_cuda


def render_deformable_cuda(
    primitive_state: dict[str, torch.Tensor] | None = None,
    canvas_hw: tuple[int, int] | int | None = None,
    height: int | torch.Tensor | None = None,
    width: int | torch.Tensor | None = None,
    deformation: torch.Tensor | None = None,
    displacement: torch.Tensor | None = None,
    deform: torch.Tensor | None = None,
    flow: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    field = deformation
    if field is None:
        field = displacement
    if field is None:
        field = deform
    if field is None:
        field = flow
    parsed_height = None
    parsed_width = None
    if isinstance(canvas_hw, (tuple, list)) and len(canvas_hw) == 2:
        parsed_height = int(canvas_hw[0])
        parsed_width = int(canvas_hw[1])
    elif isinstance(canvas_hw, int) and isinstance(height, int):
        parsed_height = int(canvas_hw)
        parsed_width = int(height)
        if torch.is_tensor(width) and field is None:
            field = width
    elif torch.is_tensor(canvas_hw) and field is None:
        field = canvas_hw
    if torch.is_tensor(height) and field is None:
        field = height
    if torch.is_tensor(width) and field is None:
        field = width
    return render_cuda(
        primitive_state=primitive_state,
        height=parsed_height,
        width=parsed_width,
        deformation_grid=field,
        deformation_enabled=field is not None or bool(kwargs.pop("deformation_enabled", False)),
        **kwargs,
    )


deformable_render_cuda = render_deformable_cuda
render_cuda_deformable = render_deformable_cuda
rasterize_deformable_cuda = render_deformable_cuda
deformable_rasterize_cuda = render_deformable_cuda
bitmap_primitives_deformable_forward_cuda = render_deformable_cuda
bitmap_primitives_forward_cuda = render_cuda
