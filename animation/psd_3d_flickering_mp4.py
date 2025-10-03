#!/usr/bin/env python3
"""
PSD 3D Camera Animation with Flickering Effect using PyVista

Combines 3D camera movement (zoom out + orbit) with flickering layer appearance.
Layers gradually appear with flickering effect while camera orbits around the scene.

Usage:
    python psd_3d_flickering_mp4.py input.psd -o output.mp4
"""

import os
import sys
import argparse
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from psd_tools import PSDImage
except ImportError:
    print("Error: psd_tools is required. Install with: pip install psd-tools")
    sys.exit(1)

try:
    import pyvista as pv
except ImportError:
    print("Error: pyvista is required. Install with: pip install pyvista")
    sys.exit(1)

try:
    from moviepy import ImageClip, concatenate_videoclips
except ImportError as e:
    print(f"Error: moviepy is required. Install with: pip install moviepy")
    print(f"Import error: {e}")
    sys.exit(1)


def extract_psd_layers_3d(psd_path, layer_start=None, layer_end=None, depth_range=(100, 1000), verbose=False):
    """Extract layers from PSD file with 3D positioning."""
    try:
        psd = PSDImage.open(psd_path)
        layers = list(psd)
        
        if verbose:
            print(f"Total layers found: {len(layers)}")
        
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
        
        layer_images = []
        layer_3d_positions = []
        canvas_size = (psd.width, psd.height)
        
        total_layers = len(selected_layers)
        min_z, max_z = depth_range
        
        for i, layer in enumerate(selected_layers):
            try:
                if hasattr(layer, 'compose'):
                    layer_image = layer.compose()
                elif hasattr(layer, 'topil'):
                    layer_image = layer.topil()
                else:
                    layer_image = layer.as_PIL()
                
                if layer_image is None:
                    if verbose:
                        print(f"  Skipped layer {i}: {layer.name} - No image data")
                    continue
                
                if layer_image.mode != 'RGBA':
                    layer_image = layer_image.convert('RGBA')
                
                original_x = layer.left
                original_y = layer.top
                
                z_depth = min_z + (i / max(1, total_layers - 1)) * (max_z - min_z)
                
                canvas_x = original_x + layer_image.size[0] / 2
                canvas_y = original_y + layer_image.size[1] / 2
                
                camera_distance_ref = 1500
                canvas_center_x = canvas_size[0] / 2
                canvas_center_y = canvas_size[1] / 2
                
                dx_2d = canvas_x - canvas_center_x
                dy_2d = canvas_y - canvas_center_y
                
                perspective_scale = (z_depth + camera_distance_ref) / camera_distance_ref
                
                world_x = canvas_center_x + dx_2d * perspective_scale
                world_y = canvas_center_y + dy_2d * perspective_scale
                world_z = z_depth
                
                depth_scale = perspective_scale
                
                layer_images.append(layer_image)
                layer_3d_positions.append((world_x, world_y, world_z, depth_scale))
                
                if verbose and i % 100 == 0:
                    print(f"  Processed {i+1}/{total_layers} layers...")
                    
            except Exception as e:
                if verbose:
                    print(f"  Skipped layer {i}: {layer.name} - {e}")
                continue
        
        if verbose:
            print(f"Successfully extracted {len(layer_images)} layers")
        
        reference_camera_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
        return layer_images, layer_3d_positions, canvas_size, reference_camera_pos
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return [], [], (0, 0), (0, 0, 0)


def create_textured_plane(image, position, scale, opacity=1.0):
    """Create a textured plane with opacity support."""
    width, height = image.size
    img_array = np.array(image)
    
    # Apply opacity to alpha channel
    if opacity < 1.0:
        img_array = img_array.copy()
        img_array[:, :, 3] = (img_array[:, :, 3] * opacity).astype(np.uint8)
    
    w = width * scale / 2
    h = height * scale / 2
    x, y, z = position
    
    plane = pv.Plane(
        center=(x, y, z),
        direction=(0, 0, 1),
        i_size=width * scale,
        j_size=height * scale,
        i_resolution=1,
        j_resolution=1
    )
    
    img_flipped = np.flipud(img_array)
    texture = pv.Texture(img_flipped)
    
    return plane, texture


def calculate_layer_opacity(frame_idx, layer_idx, total_frames, num_layers, 
                           flicker_intensity=0.3, layer_states=None):
    """
    Calculate opacity for a layer at given frame with flickering effect.
    
    States:
    - 0 (A): Transparent (opacity = 0)
    - 1 (B): Flickering (opacity varies)
    - 2 (C): Stable/Visible (opacity = 1)
    """
    if layer_states is None or layer_states[layer_idx] == 0:
        return 0.0
    elif layer_states[layer_idx] == 2:
        return 1.0
    else:  # State 1 (B) - Flickering
        # Multi-component flickering
        slow_frequency = 0.08 + (layer_idx % 10) * 0.01
        sine_component = math.sin(frame_idx * slow_frequency + layer_idx * 0.3)
        
        random.seed(frame_idx // 12 + layer_idx)
        random_component = random.uniform(-0.4, 0.4)
        
        breathing = math.sin(frame_idx * 0.03 + layer_idx * 0.7) * 0.25
        
        base_opacity = 0.5
        flicker_variation = sine_component * flicker_intensity + random_component * 0.15 + breathing
        current_opacity = base_opacity + flicker_variation
        return max(0.0, min(1.0, current_opacity))


def generate_camera_path(total_frames, camera_distance=1500,
                        target_pos=None, canvas_size=(512, 880), 
                        zoom_out_ratio=0.3, orbit_center_z=None, verbose=False):
    """Generate 3D camera movement path: zoom out then orbit."""
    if target_pos is None:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    
    camera_path = []
    tx, ty, tz = target_pos
    
    # Use orbit_center_z for orbit phase if provided (layers Z median)
    orbit_tz = orbit_center_z if orbit_center_z is not None else tz
    
    zoom_frames = int(total_frames * zoom_out_ratio)
    orbit_frames = total_frames - zoom_frames
    
    # Adjust orbit radius based on Z center shift
    z_shift = abs(orbit_tz - tz) if orbit_center_z is not None else 0
    max_distance = camera_distance * 2.0 + z_shift
    
    for frame_idx in range(total_frames):
        if frame_idx < zoom_frames:
            t_zoom = frame_idx / max(1, zoom_frames - 1) if zoom_frames > 1 else 1.0
            t_smooth = 0.5 - 0.5 * math.cos(t_zoom * math.pi)
            
            current_distance = camera_distance + (max_distance - camera_distance) * t_smooth
            camera_x = tx
            camera_y = ty
            camera_z = tz - current_distance
            
        else:
            t_orbit = (frame_idx - zoom_frames) / max(1, orbit_frames - 1)
            angle = t_orbit * 2 * math.pi
            
            # Orbit around (tx, ty, orbit_tz) - using layers Z median
            camera_x = tx + max_distance * math.sin(angle)
            camera_y = ty
            camera_z = orbit_tz - max_distance * math.cos(angle)
        
        camera_path.append(((camera_x, camera_y, camera_z), target_pos))
    
    if verbose:
        print(f"Generated zoom-out + orbit camera path with {total_frames} frames")
        print(f"  Zoom out: {zoom_frames} frames ({camera_distance} -> {max_distance:.1f})")
        print(f"  Orbit: {orbit_frames} frames at radius {max_distance:.1f}")
        if orbit_center_z is not None:
            z_shift = abs(orbit_tz - tz)
            print(f"  Orbit center Z: {orbit_center_z:.1f} (layers Z median, shift: {z_shift:.1f})")
    
    return camera_path


def calculate_fov_for_canvas(canvas_size, camera_distance):
    """Calculate FOV so that canvas fits exactly in the viewport."""
    canvas_height = canvas_size[1]
    fov_radians = 2 * math.atan(canvas_height / (2 * camera_distance))
    fov_degrees = math.degrees(fov_radians)
    return fov_degrees


def render_frame_with_opacity(layer_images, layer_3d_positions, layer_opacities,
                              camera_pos, camera_target, canvas_size, fov):
    """Render a single frame with per-layer opacity."""
    plotter = pv.Plotter(off_screen=True, window_size=canvas_size)
    
    # Add layers with their current opacity
    for i, (image, (x, y, z, scale), opacity) in enumerate(zip(layer_images, layer_3d_positions, layer_opacities)):
        if opacity <= 0:
            continue
        
        plane, texture = create_textured_plane(image, (x, y, z), scale, opacity)
        plotter.add_mesh(plane, texture=texture, opacity=1.0)
    
    plotter.camera_position = [
        camera_pos,
        camera_target,
        (0, -1, 0)
    ]
    
    plotter.camera.view_angle = fov
    plotter.set_background('white')
    
    img_array = plotter.screenshot(return_img=True, transparent_background=False)
    plotter.close()
    
    pil_image = Image.fromarray(img_array)
    return pil_image


def generate_3d_flickering_sequence(layer_images, layer_3d_positions, canvas_size, total_frames,
                                   camera_distance=1500, zoom_out_ratio=0.3, 
                                   flicker_intensity=0.3, final_display_seconds=4.0, fps=24, verbose=False):
    """Generate sequence with 3D camera movement and simple layer appearance (no flickering)."""
    num_layers = len(layer_images)
    
    # Calculate target and FOV
    if layer_3d_positions:
        center_x = sum(pos[0] for pos in layer_3d_positions) / len(layer_3d_positions)
        center_y = sum(pos[1] for pos in layer_3d_positions) / len(layer_3d_positions)
        target_pos = (center_x, center_y, 0)
    else:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    
    fov = calculate_fov_for_canvas(canvas_size, camera_distance)
    
    if verbose:
        print(f"Calculated FOV: {fov:.2f}° for canvas {canvas_size} at distance {camera_distance}")
    
    # Calculate layers Z median for orbit center
    layer_z_values = [pos[2] for pos in layer_3d_positions]
    orbit_center_z = sorted(layer_z_values)[len(layer_z_values) // 2]
    
    # Generate camera path with layers Z median as orbit center
    camera_path = generate_camera_path(
        total_frames, camera_distance, target_pos, canvas_size,
        zoom_out_ratio, orbit_center_z, verbose
    )
    
    # Simple random appearance logic (no flickering)
    # Create random permutation of layer indices
    import random
    layer_order = list(range(num_layers))
    random.shuffle(layer_order)
    
    # Calculate when each layer appears (spread evenly across frames)
    layer_appear_frame = {}
    for i, layer_idx in enumerate(layer_order):
        # Spread appearances across all frames
        appear_frame = int(i * total_frames / num_layers)
        layer_appear_frame[layer_idx] = appear_frame
    
    if verbose:
        print(f"Simple appearance animation:")
        print(f"  {num_layers} layers will appear randomly across {total_frames} frames")
        print(f"  Average: {total_frames/num_layers:.1f} frames per layer")
    
    frames = []
    
    if verbose:
        print(f"Rendering {total_frames} frames...")
    
    # Simple loop: just check if each layer should be visible
    for frame_idx in range(total_frames):
        # Calculate opacity for each layer (0 or 1, no flickering)
        layer_opacities = []
        for layer_idx in range(num_layers):
            if frame_idx >= layer_appear_frame[layer_idx]:
                opacity = 1.0  # Visible
            else:
                opacity = 0.0  # Hidden
            layer_opacities.append(opacity)
        
        # Get camera position
        (camera_pos, camera_target) = camera_path[frame_idx]
        
        # Render frame
        frame = render_frame_with_opacity(
            layer_images, layer_3d_positions, layer_opacities,
            camera_pos, camera_target, canvas_size, fov
        )
        frames.append(frame)
        
        if verbose and (frame_idx + 1) % 10 == 0:
            visible_count = sum(1 for o in layer_opacities if o > 0)
            print(f"  Frame {frame_idx + 1}/{total_frames}: Visible={visible_count}/{num_layers}")
    
    if verbose:
        print(f"\n✅ All {num_layers} layers appeared across {total_frames} frames")
    
    return frames


def create_3d_animation(frames, output_path, fps=24, verbose=False):
    """Create MP4 animation from frames."""
    if not frames:
        print("Error: No frames to process")
        return False
    
    try:
        clips = []
        
        for i, frame in enumerate(frames):
            img_array = np.array(frame)
            clip = ImageClip(img_array, duration=1.0/fps)
            clips.append(clip)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Created clip {i + 1}/{len(frames)}")
        
        if verbose:
            print(f"Writing video to: {output_path}")
        
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_path, fps=fps, logger=None)
        
        final_clip.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D camera animation with simple layer appearance from PSD using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    parser.add_argument('-o', '--output', help='Output MP4 file path')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds (default: 10.0)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--distance', type=float, default=1500, help='Initial camera distance (default: 1500)')
    parser.add_argument('--zoom-ratio', type=float, default=0.3, help='Zoom out phase ratio (default: 0.3)')
    parser.add_argument('--flicker-intensity', type=float, default=0.3, help='Flicker intensity (default: 0.3)')
    parser.add_argument('--depth-range', type=str, default='100,1000', help='Depth range (default: 100,1000)')
    parser.add_argument('--layer-start', type=int, help='Starting layer index')
    parser.add_argument('--layer-end', type=int, help='Ending layer index')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse depth range
    try:
        depth_parts = args.depth_range.split(',')
        depth_range = (float(depth_parts[0]), float(depth_parts[1]))
    except:
        print("Error: Invalid depth-range format")
        sys.exit(1)
    
    if not os.path.exists(args.psd_file):
        print(f"Error: PSD file not found: {args.psd_file}")
        sys.exit(1)
    
    if args.output is None:
        psd_path = Path(args.psd_file)
        args.output = str(psd_path.with_name(psd_path.stem + '_3d_simple.mp4'))
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating 3D camera animation with simple layer appearance:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {args.duration}s")
    print(f"  FPS: {args.fps}")
    print(f"  Camera: Zoom out ({args.zoom_ratio*100:.0f}%) + Orbit ({(1-args.zoom_ratio)*100:.0f}%)")
    
    # Extract layers
    print("\nExtracting layers...")
    layer_images, layer_3d_positions, canvas_size, reference_camera = extract_psd_layers_3d(
        args.psd_file, args.layer_start, args.layer_end, depth_range, args.verbose
    )
    
    if not layer_images:
        print("Error: No layers extracted")
        sys.exit(1)
    
    total_frames = int(args.duration * args.fps)
    
    # Generate sequence
    print(f"\nGenerating 3D simple appearance sequence with {len(layer_images)} layers...")
    frames = generate_3d_flickering_sequence(
        layer_images, layer_3d_positions, canvas_size, total_frames,
        args.distance, args.zoom_ratio, args.flicker_intensity, 4.0, args.fps, args.verbose
    )
    
    # Create animation
    print(f"\nCreating animation...")
    success = create_3d_animation(frames, args.output, fps=args.fps, verbose=args.verbose)
    
    if success:
        file_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n✅ Animation created: {args.output}")
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"\n❌ Failed to create animation")
        sys.exit(1)


if __name__ == "__main__":
    main()
