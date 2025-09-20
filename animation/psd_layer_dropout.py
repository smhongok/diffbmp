#!/usr/bin/env python3
"""
PSD Layer Dropout Tool

This script reads a PSD file, performs probabilistic dropout on layers based on their center coordinates,
and saves the filtered result as a new PSD file.

Dropout modes:
1. Linear: Dropout probability increases linearly along x/y axis (left->right, top->bottom, or both)
2. Radial: Dropout probability increases with distance from center (creates elliptical effect)

Usage:
    # Linear dropout (x-axis, left to right)
    python psd_layer_dropout.py input.psd -o output.psd --mode linear --axis x --direction positive --strength 0.8
    
    # Radial dropout (center outward)
    python psd_layer_dropout.py input.psd -o output.psd --mode radial --strength 0.6
    
    # Bidirectional linear dropout
    python psd_layer_dropout.py input.psd -o output.psd --mode linear --axis both --strength 0.5
"""

import argparse
import os
import sys
import random

import numpy as np


try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
except ImportError:
    print("Error: psd_tools is required. Install with: pip install psd-tools")
    sys.exit(1)


def get_layer_center(layer):
    """
    Calculate the center coordinates of a layer.
    
    Args:
        layer: PSD layer object
        
    Returns:
        tuple: (center_x, center_y) coordinates
    """
    try:
        left = getattr(layer, 'left', 0)
        top = getattr(layer, 'top', 0)
        width = getattr(layer, 'width', 0)
        height = getattr(layer, 'height', 0)
        
        center_x = left + width / 2
        center_y = top + height / 2
        
        return center_x, center_y
    except Exception:
        return 0, 0


def calculate_linear_dropout_prob(center_x, center_y, canvas_width, canvas_height, 
                                axis='x', direction='positive', strength=0.5):
    """
    Calculate dropout probability with ultra-aggressive curves starting from 50%.
    
    New behavior:
    - First 50%: completely safe (0% dropout)
    - 50-65%: rapid initial ramp
    - 65-80%: steep exponential increase
    - Last 20%: near-certain dropout (95%+)
    
    Args:
        center_x, center_y: Layer center coordinates
        canvas_width, canvas_height: Canvas dimensions
        axis: 'x', 'y', or 'both'
        direction: 'positive', 'negative', or 'both'
        strength: Maximum dropout probability (0.0 to 1.0)
        
    Returns:
        float: Dropout probability (0.0 to 1.0)
    """
    def steep_curve(normalized_pos, direction_type):
        """Apply ultra-aggressive curve starting from 50% position"""
        if direction_type == 'positive':
            if normalized_pos <= 0.5:
                # First 50%: completely safe
                return 0.0
            elif normalized_pos <= 0.65:
                # 50-65%: rapid initial ramp
                progress = (normalized_pos - 0.5) / 0.15  # 0 to 1
                return (progress ** 2) * (strength * 0.4)
            elif normalized_pos <= 0.8:
                # 65-80%: steep exponential increase
                progress = (normalized_pos - 0.65) / 0.15  # 0 to 1
                return (strength * 0.4) + (progress ** 2.5) * (strength * 0.5)
            else:
                # Last 20%: guaranteed 95%+ dropout
                end_progress = (normalized_pos - 0.8) / 0.2  # 0 to 1
                base_dropout = 0.95
                additional = end_progress * 0.049
                return min(0.999, base_dropout + additional)
                    
        elif direction_type == 'negative':
            # Reverse: right half safe, left half with steep dropout
            reversed_pos = 1.0 - normalized_pos
            return steep_curve(reversed_pos, 'positive')
            
        else:  # both directions
            # Center safe, edges with ultra-steep dropout
            dist_from_center = abs(normalized_pos - 0.5) * 2  # 0 to 1
            if dist_from_center <= 0.5:
                # Inner 50%: completely safe
                return 0.0
            elif dist_from_center <= 0.65:
                # 50-65%: rapid initial ramp
                progress = (dist_from_center - 0.5) / 0.15
                return (progress ** 2) * (strength * 0.4)
            elif dist_from_center <= 0.8:
                # 65-80%: steep exponential increase
                progress = (dist_from_center - 0.65) / 0.15
                return (strength * 0.4) + (progress ** 2.5) * (strength * 0.5)
            else:
                # Last 20%: guaranteed 95%+ dropout
                end_progress = (dist_from_center - 0.8) / 0.2
                base_dropout = 0.95
                additional = end_progress * 0.049
                return min(0.999, base_dropout + additional)
    
    if axis == 'x':
        normalized_pos = center_x / canvas_width
        prob = steep_curve(normalized_pos, direction)
            
    elif axis == 'y':
        normalized_pos = center_y / canvas_height
        prob = steep_curve(normalized_pos, direction)
            
    else:  # both axes
        x_normalized = center_x / canvas_width
        y_normalized = center_y / canvas_height
        
        if direction == 'positive':
            # Diagonal: top-left safe, bottom-right sparse
            diag_pos = (x_normalized + y_normalized) / 2
            prob = steep_curve(diag_pos, 'positive')
        elif direction == 'negative':
            # Reverse diagonal
            diag_pos = (x_normalized + y_normalized) / 2
            prob = steep_curve(diag_pos, 'negative')
        else:  # both directions
            # Distance from center with steep curve
            x_dist = abs(x_normalized - 0.5)
            y_dist = abs(y_normalized - 0.5)
            max_dist = max(x_dist, y_dist) * 2  # 0 to 1
            prob = steep_curve(max_dist, 'positive')
    
    return min(1.0, max(0.0, prob))


def calculate_radial_dropout_prob(center_x, center_y, canvas_width, canvas_height, 
                                strength=0.5, ellipse_ratio=1.0):
    """
    Calculate dropout probability with ultra-aggressive radial curve starting from 50%.
    
    New behavior:
    - Inner 50%: completely safe (0% dropout)
    - 50-65%: rapid initial ramp
    - 65-80%: steep exponential increase
    - Outer 20%: near-certain dropout (95%+)
    
    Args:
        center_x, center_y: Layer center coordinates
        canvas_width, canvas_height: Canvas dimensions
        strength: Maximum dropout probability (0.0 to 1.0)
        ellipse_ratio: Width/height ratio for elliptical dropout (1.0 = circle)
        
    Returns:
        float: Dropout probability (0.0 to 1.0)
    """
    # Normalize coordinates to [0, 1]
    norm_x = center_x / canvas_width
    norm_y = center_y / canvas_height
    
    # Calculate distance from center (0.5, 0.5)
    dx = (norm_x - 0.5) * 2  # [-1, 1]
    dy = (norm_y - 0.5) * 2  # [-1, 1]
    
    # Apply ellipse ratio
    dx *= ellipse_ratio
    
    # Calculate normalized distance (0 = center, 1 = edge)
    distance = np.sqrt(dx**2 + dy**2)
    normalized_distance = min(1.0, distance / np.sqrt(ellipse_ratio**2 + 1))
    
    # Apply ultra-aggressive radial curve for 50-60% dropout
    if normalized_distance <= 0.5:
        # Inner 50%: completely safe
        prob = 0.0
    elif normalized_distance <= 0.65:
        # 50-65%: rapid initial ramp
        progress = (normalized_distance - 0.5) / 0.15
        prob = (progress ** 2) * (strength * 0.4)
    elif normalized_distance <= 0.8:
        # 65-80%: steep exponential increase
        progress = (normalized_distance - 0.65) / 0.15
        prob = (strength * 0.4) + (progress ** 2.5) * (strength * 0.5)
    else:
        # Last 20%: guaranteed 95%+ dropout
        end_progress = (normalized_distance - 0.8) / 0.2
        base_dropout = 0.95
        additional = end_progress * 0.049
        prob = min(0.999, base_dropout + additional)
    
    return min(1.0, max(0.0, prob))


def apply_dropout_to_psd(psd_path, output_path, mode='linear', axis='x', direction='positive', 
                        strength=0.5, ellipse_ratio=1.0, seed=None, verbose=False):
    """
    Apply probabilistic dropout to PSD layers and save result.
    
    Args:
        psd_path: Input PSD file path
        output_path: Output PSD file path
        mode: 'linear' or 'radial'
        axis: 'x', 'y', or 'both' (for linear mode)
        direction: 'positive', 'negative', or 'both'
        strength: Maximum dropout probability (0.0 to 1.0)
        ellipse_ratio: Width/height ratio for elliptical dropout (radial mode)
        seed: Random seed for reproducibility
        verbose: Print detailed information
        
    Returns:
        bool: Success status
    """
    try:
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load PSD file
        psd = PSDImage.open(psd_path)
        canvas_width = psd.width
        canvas_height = psd.height
        
        if verbose:
            print(f"Loaded PSD: {canvas_width}x{canvas_height}, {len(psd)} layers")
        
        # Create new PSD with same dimensions
        new_psd = PSDImage.new('RGBA', (canvas_width, canvas_height), color=(0, 0, 0, 0))
        
        # Process each layer
        kept_layers = []
        dropped_layers = []
        
        for i, layer in enumerate(psd):
            try:
                # Get layer center coordinates
                center_x, center_y = get_layer_center(layer)
                
                # Calculate dropout probability
                if mode == 'linear':
                    dropout_prob = calculate_linear_dropout_prob(
                        center_x, center_y, canvas_width, canvas_height,
                        axis=axis, direction=direction, strength=strength
                    )
                elif mode == 'radial':
                    dropout_prob = calculate_radial_dropout_prob(
                        center_x, center_y, canvas_width, canvas_height,
                        strength=strength, ellipse_ratio=ellipse_ratio
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # Apply probabilistic dropout
                should_drop = random.random() < dropout_prob
                
                if should_drop:
                    dropped_layers.append((i, layer.name, center_x, center_y, dropout_prob))
                    if verbose:
                        print(f"  DROPPED Layer {i}: '{layer.name}' at ({center_x:.1f}, {center_y:.1f}) - prob: {dropout_prob:.3f}")
                else:
                    kept_layers.append((i, layer.name, center_x, center_y, dropout_prob))
                    if verbose:
                        print(f"  KEPT    Layer {i}: '{layer.name}' at ({center_x:.1f}, {center_y:.1f}) - prob: {dropout_prob:.3f}")
                    
                    # Add layer to new PSD (only if not dropped)
                    try:
                        # Simply copy the entire layer object instead of recreating
                        # This preserves all original properties including position
                        new_psd.append(layer)
                        
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Failed to copy layer {i}: {e}")
                        continue
                        
            except Exception as e:
                if verbose:
                    print(f"  ERROR   Layer {i}: Failed to process - {e}")
                continue
        
        # Save new PSD
        new_psd.save(output_path)
        
        # Print summary
        total_layers = len(kept_layers) + len(dropped_layers)
        dropout_rate = len(dropped_layers) / total_layers if total_layers > 0 else 0
        
        print(f"\n📊 Dropout Summary:")
        print(f"  Total layers: {total_layers}")
        print(f"  Kept layers:  {len(kept_layers)}")
        print(f"  Dropped layers: {len(dropped_layers)}")
        print(f"  Dropout rate: {dropout_rate:.1%}")
        print(f"  Output saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing PSD: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply probabilistic dropout to PSD layers based on center coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Linear dropout: left half safe, right half increasingly dropped
  python psd_layer_dropout.py input.psd -o output.psd --mode linear --axis x --direction positive --strength 0.8
  
  # Linear dropout: center safe, edges dropped (both directions)
  python psd_layer_dropout.py input.psd -o output.psd --mode linear --axis x --direction both --strength 0.6
  
  # Radial dropout: center safe, edges dropped (elliptical effect)
  python psd_layer_dropout.py input.psd -o output.psd --mode radial --strength 0.7
  
  # Radial dropout with elliptical shape (wider than tall)
  python psd_layer_dropout.py input.psd -o output.psd --mode radial --strength 0.5 --ellipse-ratio 1.5
  
  # Bidirectional linear dropout (both x and y axes)
  python psd_layer_dropout.py input.psd -o output.psd --mode linear --axis both --strength 0.4
        """
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    
    parser.add_argument('-o', '--output', required=True,
                       help='Output PSD file path')
    
    parser.add_argument('--mode', choices=['linear', 'radial'], default='linear',
                       help='Dropout mode: linear or radial (default: linear)')
    
    parser.add_argument('--axis', choices=['x', 'y', 'both'], default='x',
                       help='Axis for linear dropout: x, y, or both (default: x)')
    
    parser.add_argument('--direction', choices=['positive', 'negative', 'both'], default='positive',
                       help='Direction for dropout: positive, negative, or both (default: positive)')
    
    parser.add_argument('--strength', type=float, default=0.5,
                       help='Maximum dropout probability 0.0-1.0 (default: 0.5)')
    
    parser.add_argument('--ellipse-ratio', type=float, default=1.0,
                       help='Width/height ratio for radial dropout (default: 1.0)')
    
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.psd_file):
        print(f"Error: PSD file not found: {args.psd_file}")
        sys.exit(1)
    
    # Validate strength
    if not 0.0 <= args.strength <= 1.0:
        print(f"Error: Strength must be between 0.0 and 1.0, got {args.strength}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎯 PSD Layer Dropout")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Mode:   {args.mode}")
    
    if args.mode == 'linear':
        print(f"  Axis:   {args.axis}")
        print(f"  Direction: {args.direction}")
    elif args.mode == 'radial':
        print(f"  Ellipse ratio: {args.ellipse_ratio}")
    
    print(f"  Strength: {args.strength}")
    
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    
    # Apply dropout
    success = apply_dropout_to_psd(
        args.psd_file,
        args.output,
        mode=args.mode,
        axis=args.axis,
        direction=args.direction,
        strength=args.strength,
        ellipse_ratio=args.ellipse_ratio,
        seed=args.seed,
        verbose=args.verbose
    )
    
    if success:
        print(f"\n✅ Dropout applied successfully!")
    else:
        print(f"\n❌ Failed to apply dropout")
        sys.exit(1)


if __name__ == "__main__":
    main()
