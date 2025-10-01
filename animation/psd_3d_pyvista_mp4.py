#!/usr/bin/env python3
"""
PSD 3D Camera Animation with PyVista

Creates 3D camera movement animations from PSD layers using PyVista:
- Each layer is positioned in 3D space as a textured plane
- Full RGBA support with proper alpha blending
- Camera can move freely in 3D space
- Initial view matches the original PSD layout

Usage:
    python psd_3d_pyvista_mp4.py input.psd -o output.mp4
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
                
                # Calculate 3D position
                world_x = original_x + layer_image.size[0] / 2
                world_y = original_y + layer_image.size[1] / 2
                world_z = z_depth
                
                # No depth scaling - keep original size
                depth_scale = 1.0
                
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


def render_frame_pyvista(planes, camera_pos, camera_target, canvas_size, fov=60):
    """
    Render a single frame using PyVista offscreen renderer.
    
    Args:
        planes: List of (plane, texture) tuples
        camera_pos: (x, y, z) camera position
        camera_target: (x, y, z) point camera is looking at
        canvas_size: (width, height) of output
        fov: Field of view in degrees
        
    Returns:
        PIL Image of rendered frame
    """
    # Create plotter for offscreen rendering
    plotter = pv.Plotter(off_screen=True, window_size=canvas_size)
    
    # Add all planes to scene with textures
    for plane, texture in planes:
        plotter.add_mesh(plane, texture=texture, opacity=1.0)
    
    # Set camera position
    plotter.camera_position = [
        camera_pos,  # camera position
        camera_target,  # focal point
        (0, -1, 0)  # view up vector (Y down for screen coordinates)
    ]
    
    # Set field of view
    plotter.camera.view_angle = fov
    
    # Set background to white
    plotter.set_background('white')
    
    # Render to image
    img_array = plotter.screenshot(return_img=True, transparent_background=False)
    
    # Close plotter
    plotter.close()
    
    # Convert to PIL Image
    pil_image = Image.fromarray(img_array)
    
    return pil_image


def generate_camera_path(total_frames, movement_type='orbit', camera_distance=1500,
                        target_pos=None, canvas_size=(512, 880), reference_camera_pos=None,
                        verbose=False):
    """
    Generate 3D camera movement path.
    
    Args:
        total_frames: Number of frames
        movement_type: Type of movement
        camera_distance: Distance from target
        target_pos: Target position
        canvas_size: Canvas size
        reference_camera_pos: Reference camera position
        verbose: Verbose output
        
    Returns:
        List of (camera_pos, target_pos) tuples
    """
    if target_pos is None:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 500)
    
    if reference_camera_pos is None:
        reference_camera_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    
    camera_path = []
    tx, ty, tz = target_pos
    ref_x, ref_y, ref_z = reference_camera_pos
    
    for frame_idx in range(total_frames):
        t = frame_idx / max(1, total_frames - 1)
        
        if movement_type == 'orbit':
            # Circular orbit - start from reference position
            if frame_idx == 0:
                camera_x, camera_y, camera_z = ref_x, ref_y, ref_z
            else:
                angle = t * 2 * math.pi
                camera_x = tx + camera_distance * math.cos(angle)
                camera_y = ty
                camera_z = tz + camera_distance * math.sin(angle)
            
        elif movement_type == 'orbit_vertical':
            # Vertical orbit
            angle = t * 2 * math.pi
            camera_x = tx
            camera_y = ty + camera_distance * 0.5 * math.sin(angle)
            camera_z = tz + camera_distance * math.cos(angle)
            
        elif movement_type == 'spiral':
            # Spiral movement
            angle = t * 4 * math.pi
            radius = camera_distance * (1.0 - 0.3 * t)
            camera_x = tx + radius * math.cos(angle)
            camera_y = ty + camera_distance * 0.3 * math.sin(t * 2 * math.pi)
            camera_z = tz + radius * math.sin(angle)
            
        elif movement_type == 'fly_through':
            # Fly through
            camera_x = tx + camera_distance * (0.5 - t)
            camera_y = ty + camera_distance * 0.2 * math.sin(t * 4 * math.pi)
            camera_z = tz - camera_distance + t * camera_distance * 2
            
        elif movement_type == 'zoom':
            # Zoom in/out
            zoom_factor = 1.0 + 0.8 * math.sin(t * 2 * math.pi)
            camera_x = ref_x
            camera_y = ref_y
            camera_z = ref_z - camera_distance * zoom_factor
            
        else:
            # Static
            camera_x, camera_y, camera_z = ref_x, ref_y, ref_z
        
        camera_path.append(((camera_x, camera_y, camera_z), target_pos))
    
    if verbose:
        print(f"Generated {movement_type} camera path with {total_frames} frames")
    
    return camera_path


def generate_3d_sequence(layer_images, layer_3d_positions, canvas_size, total_frames,
                        movement_type='orbit', camera_distance=1500, fov=60, verbose=False):
    """
    Generate sequence of frames with 3D camera movement using PyVista.
    
    Args:
        layer_images: List of PIL Images
        layer_3d_positions: List of (x, y, z, scale) tuples
        canvas_size: (width, height)
        total_frames: Number of frames
        movement_type: Camera movement type
        camera_distance: Camera distance
        fov: Field of view
        verbose: Verbose output
        
    Returns:
        List of PIL Images
    """
    # Create 3D scene
    planes = create_3d_scene(layer_images, layer_3d_positions, verbose)
    
    # Calculate scene center
    if layer_3d_positions:
        center_x = sum(pos[0] for pos in layer_3d_positions) / len(layer_3d_positions)
        center_y = sum(pos[1] for pos in layer_3d_positions) / len(layer_3d_positions)
        center_z = sum(pos[2] for pos in layer_3d_positions) / len(layer_3d_positions)
        target_pos = (center_x, center_y, center_z)
    else:
        target_pos = (canvas_size[0] / 2, canvas_size[1] / 2, 500)
    
    # Generate camera path
    reference_camera = (canvas_size[0] / 2, canvas_size[1] / 2, 0)
    camera_path = generate_camera_path(
        total_frames, movement_type, camera_distance, target_pos, canvas_size,
        reference_camera, verbose
    )
    
    frames = []
    
    if verbose:
        print(f"Rendering {total_frames} frames...")
    
    for frame_idx, (camera_pos, camera_target) in enumerate(camera_path):
        frame = render_frame_pyvista(planes, camera_pos, camera_target, canvas_size, fov)
        frames.append(frame)
        
        if verbose and (frame_idx + 1) % 10 == 0:
            print(f"  Rendered frame {frame_idx + 1}/{total_frames}")
    
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
        description="Create 3D camera animation from PSD using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    parser.add_argument('-o', '--output', help='Output MP4 file path')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration in seconds (default: 10.0)')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second (default: 24)')
    parser.add_argument('--movement', choices=['orbit', 'orbit_vertical', 'spiral', 'fly_through', 'zoom'],
                       default='orbit', help='Camera movement (default: orbit)')
    parser.add_argument('--distance', type=float, default=1500, help='Camera distance (default: 1500)')
    parser.add_argument('--fov', type=float, default=60, help='Field of view (default: 60)')
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
        args.output = str(psd_path.with_name(psd_path.stem + '_3d_pyvista.mp4'))
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating 3D camera animation with PyVista:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {args.duration}s")
    print(f"  FPS: {args.fps}")
    print(f"  Movement: {args.movement}")
    
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
    print(f"\nGenerating 3D sequence with {len(layer_images)} layers...")
    frames = generate_3d_sequence(
        layer_images, layer_3d_positions, canvas_size, total_frames,
        args.movement, args.distance, args.fov, args.verbose
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
