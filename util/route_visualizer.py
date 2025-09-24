import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Dict, Any, Tuple

# Route visualization configuration
USE_RANGE_THRESHOLDING: bool = True  # Use trajectory range instead of length for route visualization
MOVEMENT_THRESHOLD: float = 0.0  # Minimum movement range (in pixels) to visualize a route


class RouteVisualizer:
    """
    Visualizes the movement routes of primitives across sequential frames.
    Shows how primitive positions (x, y) change between frames by drawing
    interpolated paths overlaid on the final rendered frame.
    """
    
    def __init__(self, line_width: float = 1.0, alpha: float = 0.7, 
                 max_primitives_to_show: int = 100, color_scheme: str = 'rainbow'):
        """
        Initialize the RouteVisualizer.
        
        Args:
            line_width: Width of the route lines
            alpha: Transparency of the route lines (0.0 to 1.0)
            max_primitives_to_show: Maximum number of primitives to visualize (for performance)
            color_scheme: Color scheme for routes ('rainbow', 'heat', 'cool')
        """
        self.line_width = line_width
        self.alpha = alpha
        self.max_primitives_to_show = max_primitives_to_show
        self.color_scheme = color_scheme
    
    def visualize_routes(self, frame_results: List[Dict[str, Any]], 
                        output_path: str, 
                        interpolation_points: int = 10) -> None:
        """
        Create route visualizations: one overlaid on the last rendered frame and one on white background.
        
        Args:
            frame_results: List of frame optimization results containing x, y, rendered_frame
            output_path: Path to save the visualization PNG (will create two files)
            interpolation_points: Number of interpolation points between frames
        """
        if len(frame_results) < 2:
            print("RouteVisualizer: Need at least 2 frames to visualize routes")
            return
        
        # Get the last rendered frame as background
        last_frame = frame_results[-1]['rendered_frame']
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(last_frame):
            background = last_frame.detach().cpu().numpy()
        else:
            background = last_frame
        
        # Ensure background is in correct format [H, W, 3] with values in [0, 1]
        if background.max() > 1.0:
            background = background / 255.0
        
        # Extract primitive positions from all frames
        all_positions = self._extract_positions(frame_results)
        
        # Create visualization with image background
        self._create_route_visualization(background, all_positions, output_path, interpolation_points)
        
        # Create visualization with white background
        white_bg_path = output_path.replace('.png', '_routes_only.png')
        H, W = background.shape[:2]
        white_background = np.ones((H, W, 3), dtype=np.float32)  # White background
        self._create_route_visualization(white_background, all_positions, white_bg_path, interpolation_points, white_bg=True)
    
    def _extract_positions(self, frame_results: List[Dict[str, Any]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract x, y positions from all frames.
        
        Args:
            frame_results: List of frame optimization results
            
        Returns:
            List of (x, y) position arrays for each frame
        """
        positions = []
        
        for frame_result in frame_results:
            x = frame_result['x']
            y = frame_result['y']
            
            # Convert to numpy if tensors
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            if torch.is_tensor(y):
                y = y.detach().cpu().numpy()
            
            positions.append((x, y))
        
        return positions
    
    def _create_route_visualization(self, background: np.ndarray, 
                                  positions: List[Tuple[np.ndarray, np.ndarray]], 
                                  output_path: str, 
                                  interpolation_points: int,
                                  white_bg: bool = False) -> None:
        """
        Create and save the route visualization.
        
        Args:
            background: Background image [H, W, 3]
            positions: List of (x, y) position arrays for each frame
            output_path: Path to save the visualization
            interpolation_points: Number of interpolation points between frames
            white_bg: Whether this is a white background version (affects styling)
        """
        H, W = background.shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(background, extent=[0, W, H, 0])  # Note: matplotlib uses different coordinate system
        
        # Get number of primitives and limit if necessary
        num_primitives = len(positions[0][0])
        primitives_to_show = min(num_primitives, self.max_primitives_to_show)
        
        # Select which primitives to show (evenly spaced)
        if num_primitives > self.max_primitives_to_show:
            primitive_indices = np.linspace(0, num_primitives - 1, primitives_to_show, dtype=int)
        else:
            primitive_indices = np.arange(num_primitives)
        
        # Generate colors for each primitive
        colors = self._generate_colors(len(primitive_indices))
        
        print(f"RouteVisualizer: Visualizing routes for {len(primitive_indices)} primitives across {len(positions)} frames")
        print(f"Image dimensions: H={H}, W={W}")
        
        # Debug: Check first few primitives' coordinates
        if len(positions) >= 2:
            for debug_idx in range(min(3, len(primitive_indices))):
                prim_idx = primitive_indices[debug_idx]
                x_vals = [pos[0][prim_idx] for pos in positions]
                y_vals = [pos[1][prim_idx] for pos in positions]
                print(f"Debug primitive {prim_idx}: x_raw={x_vals}, y_raw={y_vals}")
        
        # Draw routes for each selected primitive
        for i, prim_idx in enumerate(primitive_indices):
            # Extract trajectory for this primitive
            trajectory_x = []
            trajectory_y = []
            
            for frame_pos in positions:
                x_pos, y_pos = frame_pos
                trajectory_x.append(x_pos[prim_idx])
                trajectory_y.append(y_pos[prim_idx])
            
            # Coordinates are already in pixel units, just convert to numpy arrays
            trajectory_x = np.array(trajectory_x)
            trajectory_y = np.array(trajectory_y)
            
            # Calculate trajectory range for movement detection
            x_range = trajectory_x.max() - trajectory_x.min()
            y_range = trajectory_y.max() - trajectory_y.min()
            movement_range = max(x_range, y_range)  # Use the larger of x or y range
            
            # Debug output for first few primitives
            if i < 3:
                print(f"Primitive {prim_idx}: x_coords={trajectory_x}, y_coords={trajectory_y}")
                print(f"Primitive {prim_idx}: x_range=[{trajectory_x.min():.2f}, {trajectory_x.max():.2f}], y_range=[{trajectory_y.min():.2f}, {trajectory_y.max():.2f}]")
                print(f"Primitive {prim_idx}: movement_range={movement_range:.2f}, threshold={MOVEMENT_THRESHOLD}")
            
            # Adjust styling based on background type
            if white_bg:
                # For white background: thicker lines, higher alpha, black edges
                line_width = self.line_width * 2.0
                alpha = min(1.0, self.alpha * 1.3)
                edge_color = 'black'
                marker_size = 40
            else:
                # For image background: original styling
                line_width = self.line_width
                alpha = self.alpha
                edge_color = 'white'
                marker_size = 30
            
            # Only visualize primitives that meet the movement threshold
            if USE_RANGE_THRESHOLDING and movement_range >= MOVEMENT_THRESHOLD:
                # Create interpolated path and draw route
                interp_x, interp_y = self._interpolate_path(trajectory_x, trajectory_y, interpolation_points)
                
                # Draw the route line
                ax.plot(interp_x, interp_y, color=colors[i], linewidth=line_width, 
                       alpha=alpha, marker='o', markersize=3 if white_bg else 2)
                
                # Mark start point (square)
                ax.scatter(trajectory_x[0], trajectory_y[0], color=colors[i], s=marker_size, 
                          marker='s', alpha=0.9 if white_bg else 0.8, edgecolors=edge_color, linewidth=2 if white_bg else 1)
                
                # Mark end point (star)
                ax.scatter(trajectory_x[-1], trajectory_y[-1], color=colors[i], s=marker_size, 
                          marker='*', alpha=0.9 if white_bg else 0.8, edgecolors=edge_color, linewidth=2 if white_bg else 1)
        
        # Set up the plot
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Flip y-axis to match image coordinates
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust title and legend based on background type
        if white_bg:
            title_text = f'Primitive Movement Routes (Routes Only)\n({len(primitive_indices)} primitives, {len(positions)} frames)'
            title_color = 'black'
            legend_color = 'black'
        else:
            title_text = f'Primitive Movement Routes\n({len(primitive_indices)} primitives, {len(positions)} frames)'
            title_color = 'white'
            legend_color = 'gray'
        
        ax.set_title(title_text, fontsize=14, pad=20, color=title_color)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color=legend_color, linestyle='None', markersize=8, label='Start Position'),
            plt.Line2D([0], [0], marker='*', color=legend_color, linestyle='None', markersize=10, label='End Position'),
            plt.Line2D([0], [0], marker='o', color=legend_color, linestyle='None', markersize=8, label='No Movement'),
            plt.Line2D([0], [0], color=legend_color, linewidth=2, label='Movement Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Print appropriate message based on background type
        if white_bg:
            print(f"RouteVisualizer: Saved routes-only visualization to {output_path}")
        else:
            print(f"RouteVisualizer: Saved route visualization with background to {output_path}")
    
    def _interpolate_path(self, x: np.ndarray, y: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate path between waypoints.
        
        Args:
            x: X coordinates of waypoints
            y: Y coordinates of waypoints
            num_points: Number of interpolation points between each pair of waypoints
            
        Returns:
            Interpolated x and y coordinates
        """
        if len(x) < 2:
            return x, y
        
        # Create parameter array for interpolation
        t_original = np.linspace(0, 1, len(x))
        t_interp = np.linspace(0, 1, (len(x) - 1) * num_points + 1)
        
        # Interpolate x and y coordinates
        x_interp = np.interp(t_interp, t_original, x)
        y_interp = np.interp(t_interp, t_original, y)
        
        return x_interp, y_interp
    
    def _generate_colors(self, num_colors: int) -> List[str]:
        """
        Generate colors for the routes based on the selected color scheme.
        
        Args:
            num_colors: Number of colors to generate
            
        Returns:
            List of color strings
        """
        if self.color_scheme == 'rainbow':
            # Use HSV color space for rainbow colors
            colors = []
            for i in range(num_colors):
                hue = i / max(1, num_colors - 1)  # Avoid division by zero
                colors.append(plt.cm.hsv(hue))
        elif self.color_scheme == 'heat':
            # Use heat colormap
            colors = [plt.cm.hot(i / max(1, num_colors - 1)) for i in range(num_colors)]
        elif self.color_scheme == 'cool':
            # Use cool colormap
            colors = [plt.cm.cool(i / max(1, num_colors - 1)) for i in range(num_colors)]
        else:
            # Default to rainbow
            colors = [plt.cm.hsv(i / max(1, num_colors - 1)) for i in range(num_colors)]
        
        return colors


def create_route_visualization(frame_results: List[Dict[str, Any]], 
                             output_path: str,
                             line_width: float = 1.5,
                             alpha: float = 0.7,
                             max_primitives: int = 100,
                             color_scheme: str = 'rainbow',
                             interpolation_points: int = 10) -> None:
    """
    Convenience function to create route visualization.
    
    Args:
        frame_results: List of frame optimization results containing x, y, rendered_frame
        output_path: Path to save the visualization PNG
        line_width: Width of the route lines
        alpha: Transparency of the route lines
        max_primitives: Maximum number of primitives to visualize
        color_scheme: Color scheme for routes ('rainbow', 'heat', 'cool')
        interpolation_points: Number of interpolation points between frames
    """
    visualizer = RouteVisualizer(
        line_width=line_width,
        alpha=alpha,
        max_primitives_to_show=max_primitives,
        color_scheme=color_scheme
    )
    
    visualizer.visualize_routes(frame_results, output_path, interpolation_points)
