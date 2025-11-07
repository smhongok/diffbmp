"""
Transition video exporter for creating smooth interpolations between multiple images.
"""
import os
import torch
import numpy as np
from datetime import datetime
import cv2
import math


def interpolate_params(params_start, params_end, t, interpolation_style='linear'):
    """
    Interpolate between two parameter sets using different interpolation styles.
    
    Args:
        params_start: Tuple of (x, y, r, c, v, theta) tensors for start state
        params_end: Tuple of (x, y, r, c, v, theta) tensors for end state
        t: Interpolation weight (0.0 to 1.0)
        interpolation_style: Style of interpolation. Options:
            - 'linear': Constant speed (default)
            - 'ease_in': Slow start, accelerates
            - 'ease_out': Fast start, decelerates
            - 'ease_in_out': Slow start and end, fast middle
            - 'ease_in_back': Slight reverse before moving forward (anticipation)
            - 'ease_out_back': Overshoots then settles (energetic)
            - 'sine': Smooth sinusoidal easing (very natural)
            - 'exponential': Dramatic exponential acceleration/deceleration
    
    Returns:
        Tuple of interpolated (x, y, r, c, v, theta) tensors
    """
    x_start, y_start, r_start, c_start, v_start, theta_start = params_start
    x_end, y_end, r_end, c_end, v_end, theta_end = params_end
    
    # Apply easing function based on interpolation style
    if interpolation_style == 'ease_in':
        # Cubic ease-in: slow start, accelerates
        t_eased = t * t * t
        
    elif interpolation_style == 'ease_out':
        # Cubic ease-out: fast start, decelerates
        t_eased = 1 - pow(1 - t, 3)
        
    elif interpolation_style == 'ease_in_out':
        # Cubic ease-in-out: smooth acceleration and deceleration
        if t < 0.5:
            t_eased = 4 * t * t * t
        else:
            t_eased = 1 - pow(-2 * t + 2, 3) / 2
            
    elif interpolation_style == 'ease_in_back':
        # Back easing in: slight reverse before moving forward (anticipation)
        c1 = 1.70158
        t_eased = (c1 + 1) * t * t * t - c1 * t * t
        
    elif interpolation_style == 'ease_out_back':
        # Back easing out: overshoots target then settles (energetic feel)
        c1 = 1.70158
        t_eased = 1 + (c1 + 1) * pow(t - 1, 3) + c1 * pow(t - 1, 2)
        
    elif interpolation_style == 'sine':
        # Sinusoidal easing: very smooth and natural
        t_eased = -(math.cos(math.pi * t) - 1) / 2
        
    elif interpolation_style == 'exponential':
        # Exponential easing: dramatic but smooth
        if t == 0:
            t_eased = 0
        elif t == 1:
            t_eased = 1
        elif t < 0.5:
            t_eased = pow(2, 20 * t - 10) / 2
        else:
            t_eased = (2 - pow(2, -20 * t + 10)) / 2
            
    else:  # 'linear' or default
        t_eased = t
    
    # Interpolate: param_t = (1-t_eased) * start + t_eased * end
    x_t = (1 - t_eased) * x_start + t_eased * x_end
    y_t = (1 - t_eased) * y_start + t_eased * y_end
    r_t = (1 - t_eased) * r_start + t_eased * r_end
    c_t = (1 - t_eased) * c_start + t_eased * c_end
    v_t = (1 - t_eased) * v_start + t_eased * v_end
    theta_t = (1 - t_eased) * theta_start + t_eased * theta_end
    
    return x_t, y_t, r_t, c_t, v_t, theta_t


def write_frames_to_mp4(frames, video_path, fps):
    """
    Write a list of frames to an MP4 video file using OpenCV.
    
    Args:
        frames: List of numpy arrays [H, W, 3] in RGB format, uint8
        video_path: Path to save the MP4 file
        fps: Frames per second for the output video
    """
    print(f"\nWriting video to {video_path}...")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"✓ Transition video exported successfully!")
    print(f"  Path: {video_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames) / fps:.2f} seconds")


def export_transition_video_with_holds(
    x_list, y_list, r_list, c_list, v_list, theta_list,
    renderer,
    output_folder="outputs/",
    fps=30,
    transition_frames=60,
    hold_frames=30,
    interpolation_style='linear',
    device='cuda'
):
    """
    Export a transition video with hold frames at each image state.
    
    Similar to export_transition_video but adds hold_frames of static frames
    at each image before transitioning to the next.
    
    Args:
        x_list, y_list, r_list, c_list, v_list, theta_list: Lists of parameter tensors
        renderer: Renderer instance
        output_folder: Directory to save the output video
        fps: Frames per second
        transition_frames: Number of frames for each transition
        hold_frames: Number of frames to hold at each image
        interpolation_style: Interpolation style ('linear', 'ease_in_out')
        device: Device to use for computation
    
    Returns:
        str: Path to the exported video file
    """
    num_images = len(x_list)
    
    if num_images < 2:
        print("Need at least 2 images for transition. Skipping transition export.")
        return None
    
    print(f"\n{'='*80}")
    print(f"EXPORTING TRANSITION VIDEO (WITH HOLDS)")
    print(f"{'='*80}")
    print(f"Number of images: {num_images}")
    print(f"Hold frames per image: {hold_frames}")
    print(f"Transition frames per pair: {transition_frames}")
    print(f"Total frames: {num_images * hold_frames + (num_images - 1) * transition_frames}")
    print(f"FPS: {fps}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = os.path.join(output_folder, f'transition_with_holds_{timestamp}.mp4')
    
    frames = []
    
    # Process each image
    for img_idx in range(num_images):
        print(f"\nProcessing image {img_idx}...")
        
        # Get parameters for this image
        x = x_list[img_idx].to(device)
        y = y_list[img_idx].to(device)
        r = r_list[img_idx].to(device)
        c = c_list[img_idx].to(device)
        v = v_list[img_idx].to(device)
        theta = theta_list[img_idx].to(device)
        
        # Render hold frames
        print(f"  Rendering {hold_frames} hold frames...")
        with torch.no_grad():
            white_bg = torch.ones((renderer.H, renderer.W, 3), device=device)
            rendered_frame = renderer.render_from_params(
                x, y, r, theta, v, c,
                I_bg=white_bg,
                sigma=0.0,
                is_final=True
            )
            frame_np = (rendered_frame.cpu().numpy() * 255).astype(np.uint8)
        
        # Add hold frames
        for _ in range(hold_frames):
            frames.append(frame_np)
        
        # Add transition to next image (if not the last image)
        if img_idx < num_images - 1:
            print(f"  Generating transition to image {img_idx + 1}...")
            
            x_end = x_list[img_idx + 1].to(device)
            y_end = y_list[img_idx + 1].to(device)
            r_end = r_list[img_idx + 1].to(device)
            c_end = c_list[img_idx + 1].to(device)
            v_end = v_list[img_idx + 1].to(device)
            theta_end = theta_list[img_idx + 1].to(device)
            
            # Create parameter tuples once (avoid overhead in loop)
            params_start = (x, y, r, c, v, theta)
            params_end = (x_end, y_end, r_end, c_end, v_end, theta_end)
            
            for frame_idx in range(transition_frames):
                t = frame_idx / transition_frames
                
                # Interpolate all parameters using helper function
                x_t, y_t, r_t, c_t, v_t, theta_t = interpolate_params(params_start, params_end, t, interpolation_style)
                
                with torch.no_grad():
                    white_bg = torch.ones((renderer.H, renderer.W, 3), device=device)
                    rendered_frame = renderer.render_from_params(
                        x_t, y_t, r_t, theta_t, v_t, c_t,
                        I_bg=white_bg,
                        sigma=0.0,
                        is_final=True
                    )
                    frame_np = (rendered_frame.cpu().numpy() * 255).astype(np.uint8)
                    frames.append(frame_np)
    
    # Write video using helper function
    write_frames_to_mp4(frames, video_path, fps)
    
    return video_path
