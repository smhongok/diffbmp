#!/usr/bin/env python3
"""
PSD Flickering Animation Converter

This script creates a flickering animation effect from PSD layers.
Layers start transparent, some flicker with varying opacity, and gradually
more layers become permanently visible until the final image is complete.

Usage:
    python psd_flickering_mp4.py input.psd -o output.mp4 --duration 10 --fps 30
"""

import argparse
import os
import sys
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from psd_tools import PSDImage
except ImportError:
    print("Error: psd_tools is required. Install with: pip install psd-tools")
    sys.exit(1)

try:
    from moviepy import ImageClip, concatenate_videoclips
except ImportError as e:
    print(f"Error: moviepy is required. Install with: pip install moviepy")
    print(f"Import error details: {e}")
    sys.exit(1)


def extract_psd_layers(psd_path, layer_start=None, layer_end=None, verbose=False):
    """
    Extract individual layer images from PSD file (without cumulative compositing).
    
    Args:
        psd_path (str): Path to PSD file
        layer_start (int): Starting layer index (0-based, inclusive)
        layer_end (int): Ending layer index (0-based, inclusive)
        verbose (bool): Print layer information
        
    Returns:
        tuple: (list of PIL Images, canvas_size)
    """
    try:
        psd = PSDImage.open(psd_path)
        layers = list(psd)
        
        if verbose:
            print(f"Total layers found: {len(layers)}")
            for i, layer in enumerate(layers):
                print(f"  Layer {i}: {layer.name} (visible: {layer.visible})")
        
        # Apply layer range filtering
        if layer_start is not None:
            layer_start = max(0, layer_start)
        else:
            layer_start = 0
            
        if layer_end is not None:
            layer_end = min(len(layers) - 1, layer_end)
        else:
            layer_end = len(layers) - 1
            
        selected_layers = layers[layer_start:layer_end + 1]
        
        if verbose:
            print(f"Using layers {layer_start} to {layer_end} ({len(selected_layers)} layers)")
        
        # Extract individual layer images
        layer_images = []
        layer_positions = []
        canvas_size = (psd.width, psd.height)
        
        for i, layer in enumerate(selected_layers):
            try:
                # Handle different layer types
                if hasattr(layer, 'compose'):
                    layer_image = layer.compose()
                elif hasattr(layer, 'topil'):
                    layer_image = layer.topil()
                else:
                    layer_image = layer.as_PIL()
                
                # Skip if layer_image is None
                if layer_image is None:
                    if verbose:
                        print(f"  Skipping layer {layer_start + i}: {layer.name} (empty or invisible)")
                    continue
                
                # Get layer positioning information
                layer_left = getattr(layer, 'left', 0)
                layer_top = getattr(layer, 'top', 0)
                
                # Convert layer image to RGBA if necessary
                if layer_image.mode != 'RGBA':
                    layer_image = layer_image.convert('RGBA')
                
                layer_images.append(layer_image)
                layer_positions.append((layer_left, layer_top))
                
                if verbose:
                    print(f"  Extracted layer {layer_start + i}: {layer.name} (pos: {layer_left},{layer_top})")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to extract layer {layer_start + i}: {e}")
                continue
        
        return layer_images, layer_positions, canvas_size
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return [], [], (0, 0)


def create_flickering_frame(layer_images, layer_positions, canvas_size, layer_opacities):
    """
    Create a single frame with specified layer opacities.
    
    Args:
        layer_images (list): List of PIL layer images
        layer_positions (list): List of (left, top) positions for each layer
        canvas_size (tuple): (width, height) of canvas
        layer_opacities (list): Opacity values (0.0-1.0) for each layer
        
    Returns:
        PIL.Image: Composed frame
    """
    canvas = Image.new('RGBA', canvas_size, (255, 255, 255, 255))
    
    for i, (layer_image, position, opacity) in enumerate(zip(layer_images, layer_positions, layer_opacities)):
        if opacity <= 0:
            continue
            
        # Apply opacity to layer
        if opacity < 1.0:
            # Create a copy and adjust alpha
            layer_copy = layer_image.copy()
            alpha = layer_copy.split()[-1]  # Get alpha channel
            alpha = alpha.point(lambda p: int(p * opacity))  # Scale alpha
            layer_copy.putalpha(alpha)
        else:
            layer_copy = layer_image
        
        # Paste layer onto canvas
        canvas.paste(layer_copy, position, layer_copy)
    
    return canvas


def generate_flickering_sequence(layer_images, layer_positions, canvas_size, 
                               total_frames, flicker_intensity=0.3, verbose=False, final_display_seconds=4.0, fps=24):
    """
    Generate a sequence of frames with sophisticated flickering animation.
    
    Animation phases:
    1. Exponential growth: Gradually add layers to flickering state
    2. Steady stream: Maintain consistent flickering population  
    3. Drain: Transition remaining layers to stable state
    4. Final display: Show completed image for specified duration
    
    Args:
        layer_images: List of PIL Images for each layer
        layer_positions: List of (x, y) positions for each layer
        canvas_size: (width, height) of the canvas
        total_frames: Total number of frames to generate
        flicker_intensity: Maximum opacity variation (0.0-1.0)
        verbose: Print progress information
        final_display_seconds: Duration to display final completed image
        fps: Frames per second
        
    Returns:
        list: List of PIL Images for each frame
    """
    num_layers = len(layer_images)
    frames = []
    
    # Calculate final display frames
    final_display_frames = int(final_display_seconds * fps)
    transition_frames = total_frames - final_display_frames
    
    # Constants
    MAX_B_PERCENTAGE = 0.15  # Maximum 15% in B state
    max_b_count = int(num_layers * MAX_B_PERCENTAGE)
    
    # Calculate minimum frames needed for completion
    # Each layer needs time to flicker and then stabilize
    min_flicker_time = 30  # Minimum frames in B state
    transition_time_per_layer = 8  # Average frames to find high opacity for transition
    
    # Estimate minimum drain time needed
    estimated_drain_frames = max_b_count * transition_time_per_layer + min_flicker_time
    
    # Adjust phase durations based on layer count and transition frames
    if transition_frames < estimated_drain_frames * 3:
        # If transition frames is too short, extend proportionally
        growth_ratio = 0.25
        stream_ratio = 0.35
        drain_ratio = 0.40
    else:
        # Standard ratios for longer animations
        growth_ratio = 0.30
        stream_ratio = 0.45
        drain_ratio = 0.25
    
    # Phase durations (only for transition, not including final display)
    exponential_growth_frames = int(transition_frames * growth_ratio)
    steady_stream_frames = int(transition_frames * stream_ratio)
    drain_frames = transition_frames - exponential_growth_frames - steady_stream_frames
    
    if verbose:
        print(f"Generating {total_frames} frames with {num_layers} layers")
        print(f"Max B state layers: {max_b_count} ({MAX_B_PERCENTAGE*100:.1f}%)")
        print(f"Exponential growth: {exponential_growth_frames} frames")
        print(f"Steady stream: {steady_stream_frames} frames") 
        print(f"Drain phase: {drain_frames} frames")
        print(f"Final display: {final_display_frames} frames ({final_display_seconds}s)")
    
    # Initialize layer states: 0=A(transparent), 1=B(flickering), 2=C(stable)
    layer_states = [0] * num_layers
    layer_b_start_frame = [-1] * num_layers  # When each layer started flickering
    
    # Queue for B state management
    b_queue = []  # Layers currently in B state, ordered by entry time
    next_a_to_b = 0  # Next layer index to move from A to B
    
    # Calculate B state duration (how long each layer stays in B)
    b_duration = max(30, int(total_frames * 0.08))  # At least 30 frames in B state
    
    for frame_idx in range(total_frames):
        current_phase_progress = 0
        
        # Check if we're in the final display phase
        if frame_idx >= transition_frames:
            # Final display phase - transition remaining B layers to C only when opacity is high
            layers_to_graduate = []
            for layer_idx in b_queue:
                # Calculate current opacity for natural transition
                slow_frequency = 0.08 + (layer_idx % 10) * 0.01
                sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
                
                random.seed(frame_idx // 12 + layer_idx)
                random_component = random.uniform(-0.4, 0.4)
                
                breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
                
                base_opacity = 0.5
                flicker_variation = sine_component * flicker_intensity + random_component * 0.15 + breathing
                current_opacity = base_opacity + flicker_variation
                current_opacity = max(0.0, min(1.0, current_opacity))
                
                # Only graduate when opacity is naturally high (>0.85)
                if current_opacity > 0.85:
                    layers_to_graduate.append(layer_idx)
            
            # Graduate layers from B to C naturally
            for layer_idx in layers_to_graduate:
                layer_states[layer_idx] = 2  # Move to C state
                b_queue.remove(layer_idx)
            
            # Force completion in the last portion of final display
            frames_into_final = frame_idx - transition_frames
            final_completion_threshold = final_display_frames * 0.7  # Last 30% of final display
            
            if frames_into_final > final_completion_threshold and b_queue:
                # Gradually lower the opacity threshold to ensure completion
                remaining_final_frames = final_display_frames - frames_into_final
                if remaining_final_frames <= 20:  # Last 20 frames
                    # Force remaining B layers to C to ensure 100% completion
                    for layer_idx in list(b_queue):
                        layer_states[layer_idx] = 2  # Move to C state
                        b_queue.remove(layer_idx)
                elif remaining_final_frames <= 40:  # Last 40 frames
                    # Lower threshold to 0.7 for easier graduation
                    additional_graduates = []
                    for layer_idx in b_queue:
                        if layer_idx not in layers_to_graduate:  # Don't double-process
                            # Calculate opacity with same formula
                            slow_frequency = 0.08 + (layer_idx % 10) * 0.01
                            sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
                            
                            random.seed(frame_idx // 12 + layer_idx)
                            random_component = random.uniform(-0.4, 0.4)
                            
                            breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
                            
                            base_opacity = 0.5
                            flicker_variation = sine_component * flicker_intensity + random_component * 0.15 + breathing
                            current_opacity = base_opacity + flicker_variation
                            current_opacity = max(0.0, min(1.0, current_opacity))
                            
                            # Lower threshold for final completion
                            if current_opacity > 0.7:
                                additional_graduates.append(layer_idx)
                    
                    # Graduate additional layers with lower threshold
                    for layer_idx in additional_graduates:
                        layer_states[layer_idx] = 2  # Move to C state
                        b_queue.remove(layer_idx)
            
            # Gradually transition remaining A layers to B state first, then to C
            # Don't force A->C directly to avoid opacity jump
            remaining_a_layers = [i for i in range(num_layers) if layer_states[i] == 0]
            if remaining_a_layers:
                # Add a few A layers to B state each frame during final display
                layers_to_add = min(10, len(remaining_a_layers))  # Add up to 10 per frame
                for i in range(layers_to_add):
                    layer_idx = remaining_a_layers[i]
                    layer_states[layer_idx] = 1  # Move to B state first
                    layer_b_start_frame[layer_idx] = frame_idx
                    b_queue.append(layer_idx)
            
        # Determine current phase and manage state transitions (only during transition phase)
        elif frame_idx < exponential_growth_frames:
            # Phase 1: Exponential growth of B state
            phase_progress = frame_idx / exponential_growth_frames
            # Exponential curve: start with 1, grow to max_b_count
            target_b_count = max(1, int((phase_progress ** 2.5) * max_b_count))
            
            # Add layers to B state if needed
            while len(b_queue) < target_b_count and next_a_to_b < num_layers:
                layer_idx = next_a_to_b
                layer_states[layer_idx] = 1  # Move to B state
                layer_b_start_frame[layer_idx] = frame_idx
                b_queue.append(layer_idx)
                next_a_to_b += 1
                
        elif frame_idx < exponential_growth_frames + steady_stream_frames:
            # Phase 2: Steady stream - maintain 15% B state
            # Check for natural transitions when opacity is high
            layers_to_graduate = []
            for layer_idx in b_queue:
                # Calculate current opacity for this layer to check if it's naturally high
                slow_frequency = 0.08 + (layer_idx % 10) * 0.01
                sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
                
                random.seed(frame_idx // 12 + layer_idx)
                random_component = random.uniform(-0.4, 0.4)
                
                breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
                
                base_opacity = 0.5
                flicker_variation = sine_component * flicker_intensity + random_component * 0.15 + breathing
                current_opacity = base_opacity + flicker_variation
                current_opacity = max(0.0, min(1.0, current_opacity))
                
                # Only graduate if opacity is naturally high (>0.85) and minimum time has passed
                min_time_in_b = max(20, int(total_frames * 0.05))  # Minimum time in B state
                if (frame_idx - layer_b_start_frame[layer_idx] >= min_time_in_b and 
                    current_opacity > 0.85):
                    layers_to_graduate.append(layer_idx)
            
            # Graduate layers from B to C
            for layer_idx in layers_to_graduate:
                layer_states[layer_idx] = 2  # Move to C state
                b_queue.remove(layer_idx)
                
                # Add new layer to B state if available
                if next_a_to_b < num_layers:
                    new_layer_idx = next_a_to_b
                    layer_states[new_layer_idx] = 1  # Move to B state
                    layer_b_start_frame[new_layer_idx] = frame_idx
                    b_queue.append(new_layer_idx)
                    next_a_to_b += 1
                    
        else:
            # Phase 3: Drain phase - transition all remaining A layers through B to C
            layers_to_graduate = []
            for layer_idx in b_queue:
                # Calculate current opacity for natural transition
                slow_frequency = 0.08 + (layer_idx % 10) * 0.01
                sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
                
                random.seed(frame_idx // 12 + layer_idx)
                random_component = random.uniform(-0.4, 0.4)
                
                breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
                
                base_opacity = 0.5
                flicker_variation = sine_component * flicker_intensity + random_component * 0.15 + breathing
                current_opacity = base_opacity + flicker_variation
                current_opacity = max(0.0, min(1.0, current_opacity))
                
                # Graduate when opacity is naturally high
                if current_opacity > 0.85:
                    layers_to_graduate.append(layer_idx)
            
            # Graduate layers from B to C and continue adding A layers to B
            for layer_idx in layers_to_graduate:
                layer_states[layer_idx] = 2  # Move to C state
                b_queue.remove(layer_idx)
                
                # Continue adding A layers to B state during drain phase
                if next_a_to_b < num_layers:
                    new_layer_idx = next_a_to_b
                    layer_states[new_layer_idx] = 1  # Move to B state
                    layer_b_start_frame[new_layer_idx] = frame_idx
                    b_queue.append(new_layer_idx)
                    next_a_to_b += 1
            
            # If no B layers graduated but we still have A layers, force some A→B transitions
            if not layers_to_graduate and next_a_to_b < num_layers and len(b_queue) < max_b_count:
                # Add more A layers to B state to keep the process moving
                layers_to_add = min(5, max_b_count - len(b_queue), num_layers - next_a_to_b)
                for _ in range(layers_to_add):
                    if next_a_to_b < num_layers:
                        layer_idx = next_a_to_b
                        layer_states[layer_idx] = 1  # Move to B state
                        layer_b_start_frame[layer_idx] = frame_idx
                        b_queue.append(layer_idx)
                        next_a_to_b += 1
        
        # Generate layer opacities based on current states
        layer_opacities = []
        for layer_idx in range(num_layers):
            if layer_states[layer_idx] == 0:  # A state: transparent
                opacity = 0.0
            elif layer_states[layer_idx] == 2:  # C state: stable
                opacity = 1.0
            else:  # B state: flickering
                # Create flickering effect
                time_in_b = frame_idx - layer_b_start_frame[layer_idx]
                
                # Slow flickering with sine waves and randomness
                slow_frequency = 0.08 + (layer_idx % 10) * 0.01  # Vary frequency per layer
                sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
                
                # Add deterministic randomness that changes slowly
                random.seed(frame_idx // 12 + layer_idx)  # Update every 12 frames
                random_component = random.uniform(-0.4, 0.4)
                
                # Breathing effect
                breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
                
                # Smooth fade-in for new B layers to avoid opacity jump
                fade_in_duration = 20  # Frames to fade in from 0 to full flicker
                if time_in_b < fade_in_duration:
                    fade_in_factor = time_in_b / fade_in_duration
                    # Start from low opacity and gradually reach full flicker range
                    base_opacity = 0.1 + (0.4 * fade_in_factor)  # 0.1 -> 0.5
                    flicker_intensity_adjusted = flicker_intensity * fade_in_factor
                else:
                    base_opacity = 0.5
                    flicker_intensity_adjusted = flicker_intensity
                
                # Combine components
                flicker_variation = sine_component * flicker_intensity_adjusted + random_component * 0.15 * fade_in_factor + breathing * fade_in_factor
                opacity = base_opacity + flicker_variation
                opacity = max(0.0, min(1.0, opacity))  # Keep within reasonable bounds
            
            layer_opacities.append(opacity)
        
        # Create frame with current opacities
        frame = create_flickering_frame(layer_images, layer_positions, canvas_size, layer_opacities)
        frames.append(frame)
        
        if verbose and (frame_idx + 1) % 50 == 0:
            a_count = sum(1 for s in layer_states if s == 0)
            b_count = sum(1 for s in layer_states if s == 1) 
            c_count = sum(1 for s in layer_states if s == 2)
            phase_name = "Final Display" if frame_idx >= transition_frames else "Transition"
            print(f"  Frame {frame_idx + 1}/{total_frames} ({phase_name}): A={a_count} ({a_count/num_layers*100:.1f}%), "
                  f"B={b_count} ({b_count/num_layers*100:.1f}%), C={c_count} ({c_count/num_layers*100:.1f}%)")
    
    return frames


def create_flickering_animation(frames, output_path, fps=24, verbose=False):
    """
    Create MP4 animation from flickering frames.
    
    Args:
        frames (list): List of PIL Images
        output_path (str): Output MP4 file path
        fps (int): Frames per second
        verbose (bool): Print progress information
    """
    if not frames:
        print("Error: No frames to process")
        return False
    
    try:
        clips = []
        
        # Convert each frame to a short clip
        for i, frame in enumerate(frames):
            # Convert PIL Image to numpy array
            img_array = np.array(frame)
            
            # Create ImageClip with duration of 1/fps seconds
            clip = ImageClip(img_array, duration=1.0/fps)
            clips.append(clip)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Created clip {i+1}/{len(frames)}")
        
        # Concatenate clips sequentially
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Write video file
        if verbose:
            print(f"Writing video to: {output_path}")
            print(f"  Duration: {final_video.duration:.2f}s")
            print(f"  FPS: {fps}")
        
        final_video.write_videofile(
            output_path, 
            fps=fps,
            codec='libx264',
            audio=False
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return False


def calculate_optimal_duration(num_layers, fps=24, final_display_seconds=4.0):
    """
    Calculate optimal animation duration based on number of layers.
    
    Args:
        num_layers: Number of layers in the animation
        fps: Frames per second
        final_display_seconds: Duration to display final completed image
        
    Returns:
        float: Recommended duration in seconds
    """
    # Base duration for small animations (transition time only)
    base_duration = 8.0
    
    # Additional time needed based on layer count
    # More layers need more time to complete the transition process
    if num_layers <= 100:
        transition_duration = base_duration
    elif num_layers <= 500:
        transition_duration = base_duration + (num_layers - 100) * 0.01  # +1s per 100 layers
    elif num_layers <= 1000:
        transition_duration = base_duration + 4.0 + (num_layers - 500) * 0.008  # +0.8s per 100 layers
    else:
        transition_duration = base_duration + 8.0 + (num_layers - 1000) * 0.006  # +0.6s per 100 layers
    
    # Add final display time
    return transition_duration + final_display_seconds


def main():
    parser = argparse.ArgumentParser(
        description="Create flickering animation from PSD layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-calculates duration)
  python psd_flickering_mp4.py input.psd
  
  # Custom settings
  python psd_flickering_mp4.py input.psd -o flicker.mp4 --duration 8 --fps 30
  
  # Adjust flickering intensity
  python psd_flickering_mp4.py input.psd --flicker-intensity 0.5 --stabilization-rate 0.03
  
  # Use specific layer range
  python psd_flickering_mp4.py input.psd --layer-start 2 --layer-end 8
        """
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    
    parser.add_argument('-o', '--output', 
                       help='Output MP4 file path (default: input_name_flicker.mp4)')
    
    parser.add_argument('--duration', type=float, default=None,
                       help='Animation duration in seconds (default: auto-calculated based on layer count)')
    
    parser.add_argument('--fps', type=int, default=24,
                       help='Frames per second (default: 24)')
    
    parser.add_argument('--flicker-intensity', type=float, default=0.3,
                       help='Maximum flickering opacity variation (0.0-1.0, default: 0.3)')
    
    parser.add_argument('--stabilization-rate', type=float, default=0.02,
                       help='Rate at which layers stabilize (default: 0.02)')
    
    parser.add_argument('--layer-start', type=int,
                       help='Starting layer index (0-based, inclusive)')
    
    parser.add_argument('--layer-end', type=int,
                       help='Ending layer index (0-based, inclusive)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.psd_file):
        print(f"Error: PSD file not found: {args.psd_file}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        psd_path = Path(args.psd_file)
        args.output = str(psd_path.with_suffix('').with_suffix('_flicker.mp4'))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating flickering animation:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {args.duration}s")
    print(f"  FPS: {args.fps}")
    print(f"  Flicker intensity: {args.flicker_intensity}")
    print(f"  Stabilization rate: {args.stabilization_rate}")
    
    if args.layer_start is not None or args.layer_end is not None:
        print(f"  Layer range: {args.layer_start} to {args.layer_end}")
    
    # Extract layer images
    print("\nExtracting layers...")
    layer_images, layer_positions, canvas_size = extract_psd_layers(
        args.psd_file, 
        args.layer_start, 
        args.layer_end, 
        args.verbose
    )
    
    if not layer_images:
        print("Error: No layers could be extracted")
        sys.exit(1)
    
    # Auto-calculate duration if not provided
    if args.duration is None:
        args.duration = calculate_optimal_duration(len(layer_images), args.fps)
        print(f"  Auto-calculated duration: {args.duration:.1f}s (based on {len(layer_images)} layers)")
    
    # Calculate total frames
    total_frames = int(args.duration * args.fps)
    
    # Generate flickering sequence
    print(f"\nGenerating flickering sequence with {len(layer_images)} layers...")
    frames = generate_flickering_sequence(
        layer_images, 
        layer_positions, 
        canvas_size,
        total_frames,
        args.flicker_intensity,
        args.verbose,
        final_display_seconds=6.0,
        fps=args.fps
    )
    
    # Create animation
    print(f"\nCreating animation with {len(frames)} frames...")
    success = create_flickering_animation(
        frames, 
        args.output, 
        fps=args.fps,
        verbose=args.verbose
    )
    
    if success:
        print(f"\n✅ Flickering animation created successfully: {args.output}")
        
        # Show file info
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
    else:
        print("\n❌ Failed to create animation")
        sys.exit(1)


if __name__ == "__main__":
    main()
