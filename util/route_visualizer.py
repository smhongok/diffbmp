import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Dict, Any, Tuple


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
        Create route visualization overlaid on the last rendered frame.
        
        Args:
            frame_results: List of frame optimization results containing x, y, rendered_frame
            output_path: Path to save the visualization PNG
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
        
        # Create the visualization
        self._create_route_visualization(background, all_positions, output_path, interpolation_points)
    
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
                                  interpolation_points: int) -> None:
        """
        Create and save the route visualization.
        
        Args:
            background: Background image [H, W, 3]
            positions: List of (x, y) position arrays for each frame
            output_path: Path to save the visualization
            interpolation_points: Number of interpolation points between frames
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
        
        # Draw routes for each selected primitive
        for i, prim_idx in enumerate(primitive_indices):
            # Extract trajectory for this primitive
            trajectory_x = []
            trajectory_y = []
            
            for frame_pos in positions:
                x_pos, y_pos = frame_pos
                trajectory_x.append(x_pos[prim_idx])
                trajectory_y.append(y_pos[prim_idx])
            
            # Convert to image coordinates (x stays same, y needs conversion)
            trajectory_x = np.array(trajectory_x) * W  # Scale to image width
            trajectory_y = np.array(trajectory_y) * H  # Scale to image height
            
            # Create interpolated path
            if len(trajectory_x) > 1:
                interp_x, interp_y = self._interpolate_path(trajectory_x, trajectory_y, interpolation_points)
                
                # Draw the route
                ax.plot(interp_x, interp_y, color=colors[i], linewidth=self.line_width, 
                       alpha=self.alpha, marker='o', markersize=2)
                
                # Mark start and end points
                ax.scatter(trajectory_x[0], trajectory_y[0], color=colors[i], s=30, 
                          marker='s', alpha=0.8, edgecolors='white', linewidth=1)  # Start: square
                ax.scatter(trajectory_x[-1], trajectory_y[-1], color=colors[i], s=30, 
                          marker='*', alpha=0.8, edgecolors='white', linewidth=1)  # End: star
        
        # Set up the plot
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Flip y-axis to match image coordinates
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Primitive Movement Routes\n({len(primitive_indices)} primitives, {len(positions)} frames)', 
                    fontsize=14, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='gray', linestyle='None', markersize=8, label='Start Position'),
            plt.Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=10, label='End Position'),
            plt.Line2D([0], [0], color='gray', linewidth=2, label='Movement Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"RouteVisualizer: Saved route visualization to {output_path}")
    
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
