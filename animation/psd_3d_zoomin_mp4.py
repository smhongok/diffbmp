#!/usr/bin/env python3
"""
PSD 3D Camera Zoom-In Animation with PyVista

Creates 3D camera zoom-in animations from PSD layers using PyVista:
- Each layer is positioned in 3D space as a textured plane
- Full RGBA support with proper alpha blending
- Camera zooms in past all layers until nothing is visible
- Initial view matches the original PSD layout

Usage:
    python psd_3d_zoomin_mp4.py input.psd -o output.mp4
"""

import os
import sys
import argparse
import math
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
    """
    Extract layers from PSD file with 3D positioning.
    
    Args:
        psd_path: Path to PSD file
        layer_start: Starting layer index (0-based, inclusive)
        layer_end: Ending layer index (0-based, inclusive)
        depth_range: (min_z, max_z) depth range for layer positioning
        verbose: Print extraction details
        
    Returns:
        tuple: (layer_images, layer_3d_positions, canvas_size, reference_camera_pos)
    """
    try:
        psd = PSDImage.open(psd_path)
        layers = list(psd)
        
        if verbose:
            print(f"Total layers found: {len(layers)}")
        
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
        layer_3d_positions = []
        canvas_size = (psd.width, psd.height)
        
        total_layers = len(selected_layers)
        min_z, max_z = depth_range
        
        # Reference camera position to match original PSD view
        reference_camera_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
        
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
                        print(f"  Skipped layer {i}: {layer.name} - No image data")
                    continue
                
                # Convert to RGBA if needed
                if layer_image.mode != 'RGBA':
                    layer_image = layer_image.convert('RGBA')
                
                # Get original 2D position
                original_x = layer.left
                original_y = layer.top
                
                # Calculate 3D depth based on layer order
                z_depth = min_z + (i / max(1, total_layers - 1)) * (max_z - min_z)
                
                # Calculate 3D position with perspective correction
                # Original 2D position in canvas
                canvas_x = original_x + layer_image.size[0] / 2
                canvas_y = original_y + layer_image.size[1] / 2
                
                # We want the camera at (canvas_center_x, canvas_center_y, -camera_distance)
                # to see layers at their original 2D positions
                # Using perspective projection: scale = (z + camera_distance) / camera_distance
                camera_distance_ref = 1500  # Reference camera distance
                
                # Center of canvas (target point)
                canvas_center_x = canvas_size[0] / 2
                canvas_center_y = canvas_size[1] / 2
                
                # Distance from canvas center in 2D
                dx_2d = canvas_x - canvas_center_x
                dy_2d = canvas_y - canvas_center_y
                
                # Perspective scale factor
                perspective_scale = (z_depth + camera_distance_ref) / camera_distance_ref
                
                # Apply perspective to position
                world_x = canvas_center_x + dx_2d * perspective_scale
                world_y = canvas_center_y + dy_2d * perspective_scale
                world_z = z_depth
                
                # Scale layer size by perspective
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
            print(f"Canvas size: {canvas_size}")
            print(f"Reference camera: {reference_camera_pos}")
        
        return layer_images, layer_3d_positions, canvas_size, reference_camera_pos
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return [], [], (0, 0), (0, 0, 0)


def create_textured_plane(image, position, scale):
    """
    Create a textured plane for a layer image with full RGBA support.
    
    Args:
        image: PIL Image (RGBA)
        position: (x, y, z) center position in world space
        scale: Scale factor for the plane
        
    Returns:
        pyvista.PolyData plane with texture
    """
    width, height = image.size
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create a plane mesh
    w = width * scale / 2
    h = height * scale / 2
    x, y, z = position
    
    # Create plane centered at position
    plane = pv.Plane(
        center=(x, y, z),
        direction=(0, 0, 1),
        i_size=width * scale,
        j_size=height * scale,
        i_resolution=1,
        j_resolution=1
    )
    
    # Flip image vertically for correct texture mapping
    img_flipped = np.flipud(img_array)
    
    # Create texture from image (PyVista supports RGBA)
    texture = pv.Texture(img_flipped)
    
    return plane, texture


def create_3d_scene(layer_images, layer_3d_positions, verbose=False):
    """
    Create PyVista scene with all layers as textured planes.
    
    Args:
        layer_images: List of PIL Images
        layer_3d_positions: List of (x, y, z, scale) tuples
        verbose: Print progress
        
    Returns:
        List of (plane, texture) tuples
    """
    planes = []
    
    if verbose:
        print(f"Creating 3D scene with {len(layer_images)} layers...")
    
    for i, (image, (x, y, z, scale)) in enumerate(zip(layer_images, layer_3d_positions)):
        plane, texture = create_textured_plane(image, (x, y, z), scale)
        planes.append((plane, texture))
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Created plane {i+1}/{len(layer_images)}")
    
    return planes


def render_frame_pyvista(planes, camera_pos, camera_target, canvas_size, fov=60, layer_opacities=None):
    """
    Render a single frame using PyVista offscreen renderer.
    
    Args:
        planes: List of (plane, texture) tuples
        camera_pos: (x, y, z) camera position
        camera_target: (x, y, z) point camera is looking at
        canvas_size: (width, height) of output
        fov: Field of view in degrees
        layer_opacities: Optional list of opacity values per layer (0.0-1.0)
        
    Returns:
        PIL Image of rendered frame
    """
    # Create plotter for offscreen rendering
    plotter = pv.Plotter(off_screen=True, window_size=canvas_size)
    
    # Add all planes to scene with textures
    for i, (plane, texture) in enumerate(planes):
        opacity = 1.0
        if layer_opacities is not None and i < len(layer_opacities):
            opacity = layer_opacities[i]
        
        if opacity > 0:  # Only add visible layers
            plotter.add_mesh(plane, texture=texture, opacity=opacity)
    
    # Set camera position
    plotter.camera_position = [
        camera_pos,  # camera position
        camera_target,  # focal point
        (0, -1, 0)  # view up vector (Y down for screen coordinates)
    ]
    
    # Set field of view
    plotter.camera.view_angle = fov
    
    # Set clipping range to accommodate far objects (like background layer)
    # Default is auto, but we need to ensure far background is not clipped
    plotter.camera.clipping_range = (1.0, 10000.0)
    
    # Set background to white
    plotter.set_background('white')
    
    # Render to image
    img_array = plotter.screenshot(return_img=True, transparent_background=False)
    
    # Close plotter
    plotter.close()
    
    # Convert to PIL Image
    pil_image = Image.fromarray(img_array)
    
    return pil_image


def generate_zoomin_camera_path(total_frames, camera_distance=1500,
                                target_pos=None, canvas_size=(512, 880), 
                                max_depth=1000, verbose=False):
    """
    Generate 3D camera zoom-in path: camera moves forward past all layers.
    
    Args:
        total_frames: Number of frames
        camera_distance: Initial camera distance
        target_pos: Target position (center of scene)
        canvas_size: Canvas size
        max_depth: Maximum depth of layers (to zoom past them)
        verbose: Verbose output
        
    Returns:
        List of (camera_pos, target_pos) tuples
    """
    if target_pos is None:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    
    camera_path = []
    tx, ty, tz = target_pos
    
    # Camera moves from z=-1500 to z=+1500
    start_camera_z = -1500
    end_camera_z = 1500
    
    if verbose:
        print(f"Camera zoom-in path:")
        print(f"  Camera x,y: ({canvas_size[0]/2}, {canvas_size[1]/2}) (canvas center)")
        print(f"  Camera moves from z={start_camera_z} to z={end_camera_z}")
        print(f"  Camera always looks toward +z infinity (fixed direction)")
        print(f"  Total travel: {end_camera_z - start_camera_z}")
        print(f"  Layer depth range: 0 to {max_depth}")
    
    for frame_idx in range(total_frames):
        # Linear interpolation with smooth easing
        t = frame_idx / max(1, total_frames - 1)
        
        # Smooth easing: slow start, fast middle, slow end
        t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)
        
        # Interpolate camera z from -1500 to +1500
        camera_z = start_camera_z + (end_camera_z - start_camera_z) * t_smooth
        
        # Keep camera at canvas center (not layer center)
        camera_x = canvas_size[0] / 2
        camera_y = canvas_size[1] / 2
        
        # Target is always far ahead in +z direction (looking toward +infinity)
        # Set target at camera_z + large offset to ensure camera always looks forward
        forward_target = (camera_x, camera_y, camera_z + 10000)
        camera_path.append(((camera_x, camera_y, camera_z), forward_target))
    
    if verbose:
        print(f"Generated zoom-in camera path with {total_frames} frames")
        print(f"  Frame 0: camera_z={camera_path[0][0][2]:.1f}")
        print(f"  Frame {total_frames//2}: camera_z={camera_path[total_frames//2][0][2]:.1f}")
        print(f"  Frame {total_frames-1}: camera_z={camera_path[-1][0][2]:.1f}")
    
    return camera_path


def calculate_fov_for_canvas(canvas_size, camera_distance):
    """
    Calculate FOV so that canvas fits exactly in the viewport.
    
    Args:
        canvas_size: (width, height) of canvas
        camera_distance: Distance from camera to target
        
    Returns:
        FOV in degrees
    """
    # Use the larger dimension to ensure full canvas is visible
    canvas_height = canvas_size[1]
    
    # Calculate FOV using: tan(fov/2) = (height/2) / distance
    fov_radians = 2 * math.atan(canvas_height / (2 * camera_distance))
    fov_degrees = math.degrees(fov_radians)
    
    return fov_degrees


def generate_3d_sequence(layer_images, layer_3d_positions, canvas_size, total_frames,
                        camera_distance=1500, depth_range=(100, 1000), 
                        flicker_duration=3.0, fps=24, bg_layer_idx=None, bg_z_offset=2000, 
                        bg_fade=False, verbose=False):
    """
    Generate sequence of frames with 3D camera zoom-in using PyVista.
    
    Args:
        layer_images: List of PIL Images
        layer_3d_positions: List of (x, y, z, scale) tuples
        canvas_size: (width, height)
        total_frames: Number of frames
        camera_distance: Initial camera distance
        depth_range: (min_z, max_z) depth range of layers
        flicker_duration: Duration of flickering phase in seconds (0 to disable)
        fps: Frames per second
        bg_layer_idx: Index of background layer to keep always visible (None to disable)
        bg_z_offset: Z offset for background layer (default: 2000, far back)
        bg_fade: If True, background opacity fades linearly during camera movement (1.0 -> 0.0)
        verbose: Verbose output
        
    Returns:
        List of PIL Images
    """
    num_layers = len(layer_images)
    
    # Move background layer to far back if specified
    if bg_layer_idx is not None and 0 <= bg_layer_idx < num_layers:
        x, y, z, scale = layer_3d_positions[bg_layer_idx]
        new_z = z + bg_z_offset
        
        # Scale up proportionally to maintain same apparent size at original position
        # Using perspective: apparent_size = actual_size * camera_distance / (camera_distance + z)
        # We want: new_scale * camera_distance / (camera_distance + new_z) = scale * camera_distance / (camera_distance + z)
        # So: new_scale = scale * (camera_distance + new_z) / (camera_distance + z)
        scale_factor = (camera_distance + new_z) / (camera_distance + z)
        new_scale = scale * scale_factor
        
        layer_3d_positions[bg_layer_idx] = (x, y, new_z, new_scale)
        if verbose:
            print(f"Background layer {bg_layer_idx}:")
            print(f"  z: {z:.1f} -> {new_z:.1f} (offset: +{bg_z_offset})")
            print(f"  scale: {scale:.3f} -> {new_scale:.3f} (factor: {scale_factor:.3f}x)")
    
    # Create 3D scene
    planes = create_3d_scene(layer_images, layer_3d_positions, verbose)
    
    # Calculate scene center
    # Fix target at z=0 to match the camera_distance_ref assumption
    if layer_3d_positions:
        center_x = sum(pos[0] for pos in layer_3d_positions) / len(layer_3d_positions)
        center_y = sum(pos[1] for pos in layer_3d_positions) / len(layer_3d_positions)
        # Force z=0 so camera at z=-camera_distance matches perspective calculation
        target_pos = (center_x, center_y, 0)
    else:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    
    # Calculate FOV to fit canvas exactly at initial distance
    fov = calculate_fov_for_canvas(canvas_size, camera_distance)
    
    if verbose:
        print(f"Calculated FOV: {fov:.2f}° for canvas {canvas_size} at distance {camera_distance}")
    
    # Generate camera path (zoom in past all layers)
    max_depth = depth_range[1]
    camera_path = generate_zoomin_camera_path(
        total_frames, camera_distance, target_pos, canvas_size,
        max_depth, verbose
    )
    
    # Calculate flickering phase frames
    flicker_frames = int(flicker_duration * fps)
    
    # Setup random layer appearance order (excluding background)
    import random
    layer_order = [i for i in range(num_layers) if i != bg_layer_idx]
    random.shuffle(layer_order)
    
    layer_appear_frame = {}
    for i, layer_idx in enumerate(layer_order):
        appear_frame = int(i * flicker_frames / len(layer_order))
        layer_appear_frame[layer_idx] = appear_frame
    
    # Background layer always visible from frame 0
    if bg_layer_idx is not None:
        layer_appear_frame[bg_layer_idx] = 0
    
    if verbose and flicker_frames > 0:
        print(f"Flickering phase: {flicker_frames} frames ({flicker_duration}s)")
        if bg_layer_idx is not None:
            print(f"  {len(layer_order)} layers will appear randomly (bg layer {bg_layer_idx} always visible)")
            if bg_fade:
                print(f"  Background opacity: 0.5 (flicker) -> 0.0 (end of camera)")
        else:
            print(f"  {num_layers} layers will appear randomly")
    
    frames = []
    total_sequence_frames = flicker_frames + total_frames
    
    if verbose:
        print(f"Rendering {total_sequence_frames} frames ({flicker_frames} flicker + {total_frames} camera)...")
    
    for frame_idx in range(total_sequence_frames):
        # Determine layer opacities
        layer_opacities = []
        
        if frame_idx < flicker_frames:
            # Flickering phase: gradually show layers
            for layer_idx in range(num_layers):
                if frame_idx >= layer_appear_frame[layer_idx]:
                    # Background starts at 0.5 if fade is enabled
                    if layer_idx == bg_layer_idx and bg_fade:
                        opacity = 0.98
                    else:
                        opacity = 0.98
                else:
                    opacity = 0.0
                layer_opacities.append(opacity)
            
            # Camera stays at initial position during flicker
            camera_pos, camera_target = camera_path[0]
        else:
            # Camera movement phase
            path_idx = frame_idx - flicker_frames
            
            # Calculate background fade if enabled
            if bg_fade and bg_layer_idx is not None:
                # Fade from 0.5 to 0.0 over camera movement
                bg_opacity = 0.98 * (1.0 - (path_idx / max(1, total_frames - 1)))
            else:
                bg_opacity = 1.0
            
            # All layers fully visible during camera movement (except background if fading)
            for layer_idx in range(num_layers):
                if layer_idx == bg_layer_idx and bg_fade:
                    layer_opacities.append(bg_opacity)
                else:
                    layer_opacities.append(1.0)
            
            # Use camera path (offset by flicker_frames)
            camera_pos, camera_target = camera_path[path_idx]
        
        # Render frame
        frame = render_frame_pyvista(planes, camera_pos, camera_target, canvas_size, fov, layer_opacities)
        frames.append(frame)
        
        # Debug: always print transition frames
        is_transition = (frame_idx >= flicker_frames - 2 and frame_idx <= flicker_frames + 2)
        
        if verbose and ((frame_idx + 1) % 10 == 0 or is_transition):
            visible = sum(1 for o in layer_opacities if o > 0)
            phase = "flicker" if frame_idx < flicker_frames else "camera"
            info = f"  Frame {frame_idx + 1}/{total_sequence_frames} ({phase}): visible={visible}/{num_layers}, camera_z={camera_pos[2]:.1f}"
            if bg_fade and bg_layer_idx is not None:
                info += f", bg_opacity={layer_opacities[bg_layer_idx]:.2f}"
            if is_transition:
                info += " [TRANSITION]"
            print(info)
    
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
        description="Create 3D camera zoom-in animation from PSD using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    parser.add_argument('-o', '--output', help='Output MP4 file path')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds (default: 10.0)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--distance', type=float, default=1500, help='Initial camera distance (default: 1500)')
    parser.add_argument('--depth-range', type=str, default='100,1000', help='Depth range (default: 100,1000)')
    parser.add_argument('--flicker-duration', type=float, default=3.0, help='Flickering phase duration in seconds (default: 3.0, 0 to disable)')
    parser.add_argument('--bg-layer', type=int, help='Background layer index to keep always visible and move far back (e.g., 0 for first layer)')
    parser.add_argument('--bg-z-offset', type=float, default=2000, help='Z offset for background layer (default: 2000)')
    parser.add_argument('--bg-fade', action='store_true', help='Fade out background linearly during camera movement (1.0 -> 0.0)')
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
        args.output = str(psd_path.with_name(psd_path.stem + '_3d_zoomin.mp4'))
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating 3D camera zoom-in animation with PyVista:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {args.duration}s")
    print(f"  FPS: {args.fps}")
    print(f"  Camera: Zoom in past all layers")
    
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
    print(f"\nGenerating 3D zoom-in sequence with {len(layer_images)} layers...")
    if args.flicker_duration > 0:
        print(f"  Flickering phase: {args.flicker_duration}s")
    if args.bg_layer is not None:
        fade_info = " with fade" if args.bg_fade else ""
        print(f"  Background layer: {args.bg_layer} (always visible, z offset: +{args.bg_z_offset}{fade_info})")
    frames = generate_3d_sequence(
        layer_images, layer_3d_positions, canvas_size, total_frames,
        args.distance, depth_range, args.flicker_duration, args.fps, 
        args.bg_layer, args.bg_z_offset, args.bg_fade, args.verbose
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
