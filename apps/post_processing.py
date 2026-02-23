"""
Post-processing effects for circle_art.
"""

import torch
import numpy as np
import cv2
import math

# ===================================================
# Speed Curves (Easing Functions)
# ===================================================
# ── Ease-in (slow start) ──

def _ease_in_quadratic(t: torch.Tensor) -> torch.Tensor:
    """Slow start (quadratic): starts slowly, accelerates."""
    return t * t


def _ease_in_cubic(t: torch.Tensor) -> torch.Tensor:
    """Slow start (cubic): starts very slowly, accelerates harder."""
    return t * t * t


def _ease_in_sine(t: torch.Tensor) -> torch.Tensor:
    """Slow start (sine): smooth slow start."""
    return 1.0 - torch.cos(t * (math.pi / 2.0))


# ── Ease-out (slow end) ──

def _ease_out_quadratic(t: torch.Tensor) -> torch.Tensor:
    """Slow end (quadratic): decelerates to a stop."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_out_cubic(t: torch.Tensor) -> torch.Tensor:
    """Slow end (cubic): decelerates more gently."""
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


# ── Ease-in-out (slow start + slow end) ──

def _ease_in_out_quadratic(t: torch.Tensor) -> torch.Tensor:
    """Slow start and slow end (quadratic): smooth S-curve."""
    return torch.where(t < 0.5, 2.0 * t * t, 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0)


def _ease_in_out_cubic(t: torch.Tensor) -> torch.Tensor:
    """Slow start and slow end (cubic): pronounced S-curve."""
    return torch.where(t < 0.5, 4.0 * t * t * t, 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)


def _ease_in_out_sine(t: torch.Tensor) -> torch.Tensor:
    """Slow start and slow end (sine): gentle S-curve."""
    return -(torch.cos(math.pi * t) - 1.0) / 2.0


EASING_FUNCTIONS = {
    # ease-in (slow start only)
    "ease_in_quadratic": _ease_in_quadratic,
    "ease_in_cubic": _ease_in_cubic,
    "ease_in_sine": _ease_in_sine,
    # ease-out (slow end only)
    "ease_out_quadratic": _ease_out_quadratic,
    "ease_out_cubic": _ease_out_cubic,
    # ease-in-out (slow start + slow end)
    "ease_in_out_quadratic": _ease_in_out_quadratic,
    "ease_in_out_cubic": _ease_in_out_cubic,
    "ease_in_out_sine": _ease_in_out_sine,
}

# ===================================================
# Main Effect Functions
# ===================================================
def blossom_effect(
    renderer,
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    v: torch.Tensor,
    c: torch.Tensor,
    I_bg: torch.Tensor,
    output_path: str,
    fps: int = 30,
    total_duration: float = 6.0,
    anim_duration: float = 2.0,
    stagger_duration: float = 4.0,
    rotation_offset_degrees: float = 2.0,
    easing: str = "ease_in_out_cubic",
    rotation_mod: int = 77,
    hold_final_seconds: float = 1.0,
):
    """
    Blossom effect: primitives bloom into view with staggered scale and rotation.

    Blossom effect grows each primitive from a smaller scale to its final size while also rotating it into place. Each animation is staggered in time based on the primitive index, creating a realistic bloom effect across the image. 

    * Figure 1 (c) of our paper uses JSX script on Adobe After Effects to achieve a similar effect, which served as inspiration for this implementation. 

    Args (Necessary):
        renderer: SimpleTileRenderer instance for rendering frames.
        x, y: (N,) primitive positions (unchanged during animation).
        r: (N,) final primitive scales.
        theta: (N,) final primitive rotations (radians).
        v: (N,) final visibility logits.
        c: (N, 3) final color logits.
        I_bg: (H, W, 3) background tensor.
        output_path: Base PNG output path; MP4 path derived by replacing extension.

    Args (Optional, Fixed):   
        fps: Frames per second for the output video.
        total_duration: Total animation duration in seconds (excluding hold).
        anim_duration: Duration of each primitive's scale+rotation animation (seconds).
        stagger_duration: Maximum stagger offset across primitives (seconds).
        rotation_offset_degrees: Per-group rotation offset in degrees.
        easing: Easing function name. One of EASING_FUNCTIONS for smooth effect.
        rotation_mod : Modulo for grouping primitives for rotation offsets.
        hold_final_seconds: Hold the final fully-rendered frame for this many seconds.

    Returns:
        None. Saves blossom animation MP4 to output_path with '_blossom.mp4' suffix.
    """
    N = len(x)
    device = x.device

    # stagger group count based on primitive count N
    stagger_mod = max(2, N // 20)

    # Select easing function
    ease_fn = EASING_FUNCTIONS.get(easing, _ease_in_out_cubic)

    # Compute per-primitive stagger offsets 
    # Higher-index primitives bloom first (reversed order)
    indices = torch.arange(N, device=device, dtype=torch.float32)
    stagger_offsets = (
        ((N - indices) % stagger_mod) / max(stagger_mod - 1, 1)
    ) * stagger_duration

    # Compute per-primitive rotation offsets
    rotation_offsets_rad = (
        -(indices % rotation_mod) * rotation_offset_degrees * (math.pi / 180.0)
    )

    print(f"  Primitives: {N}, stagger_mod={stagger_mod}, rotation_mod={rotation_mod}")

    # Store final parameter values
    r_final = r.clone()
    theta_final = theta.clone()

    # Prepare video writer
    # mp4v
    video_path = output_path.replace('.png', '_blossom.mp4')
    H, W = renderer.H, renderer.W
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

    if not video_writer.isOpened():
        print(f"Warning: Could not open video writer for {video_path}")
        with torch.no_grad():
            rendered = renderer.render_from_params(
                x, y, r_final, theta_final, v, c,
                I_bg=I_bg, sigma=0.0, is_final=True,
            )
        return rendered

    anim_frames = int(fps * total_duration)
    hold_frames = int(fps * hold_final_seconds)
    total_frames = anim_frames + hold_frames

    print(f"Generating blossom animation: {total_frames} frames "
          f"({anim_frames} anim + {hold_frames} hold) at {fps}fps")
    print(f"  Easing: {easing} (slow start)")
    print(f"  Stagger: {stagger_duration}s over {stagger_mod} groups, "
          f"anim duration: {anim_duration}s per primitive")

    # Set up autocast context for FP16 renderers
    use_autocast = getattr(renderer, 'use_fp16', False) and torch.cuda.is_available()
    if use_autocast:
        try:
            from torch.amp import autocast
            autocast_ctx = autocast(device_type='cuda')
        except ImportError:
            from torch.cuda.amp import autocast
            autocast_ctx = autocast()
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()

    final_frame_np = None

    with torch.no_grad(), autocast_ctx:
        for frame_idx in range(total_frames):
            t_global = frame_idx / fps  # current time in seconds

            if frame_idx < anim_frames:
                # -- Per-primitive normalized progress [0, 1] --
                t_local = t_global - stagger_offsets  # (N,)
                progress_linear = torch.clamp(t_local / anim_duration, 0.0, 1.0)

                # Apply slow-start easing
                progress = ease_fn(progress_linear)

                # Interpolate scale: 0 -> r_final
                r_current = r_final * progress

                # Interpolate rotation
                # (theta_final + rotation_offset)  ->  theta_final
                theta_current = theta_final + rotation_offsets_rad * (1.0 - progress)

                rendered = renderer.render_from_params(
                    x, y, r_current, theta_current, v, c,
                    I_bg=I_bg, sigma=0.0, is_final=True,
                )
            else:
                # Hold frames: reuse cached final frame
                if final_frame_np is not None:
                    video_writer.write(
                        cv2.cvtColor(final_frame_np, cv2.COLOR_RGB2BGR)
                    )
                    if (frame_idx + 1) % fps == 0:
                        print(f"  Frame {frame_idx + 1}/{total_frames} "
                              f"({t_global:.1f}s) [hold]")
                    continue
                rendered = renderer.render_from_params(
                    x, y, r_final, theta_final, v, c,
                    I_bg=I_bg, sigma=0.0, is_final=True,
                )

            # Convert to uint8 numpy (float() needed for FP16 compatibility)
            frame_np = (rendered.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)

            # Cache the final animation frame for the hold period
            if frame_idx == anim_frames - 1 or frame_idx >= anim_frames:
                final_frame_np = frame_np

            video_writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

            if (frame_idx + 1) % fps == 0:
                print(f"  Frame {frame_idx + 1}/{total_frames} ({t_global:.1f}s)")

    video_writer.release()
    print(f"Saved blossom animation to: {video_path}")