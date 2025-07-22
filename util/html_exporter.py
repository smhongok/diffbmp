import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import base64
import io
from PIL import Image

class HTMLExporter:
    """
    Exports optimized primitives (SVG + raster) to interactive HTML with animation support.
    This replaces PDFExporter for mixed primitive types.
    """
    
    def __init__(self, primitive_loader, canvas_size: tuple, alpha_upper_bound: float = 1.0):
        self.primitive_loader = primitive_loader
        self.canvas_w, self.canvas_h = canvas_size
        self.alpha_upper_bound = alpha_upper_bound
    
    def export(self, 
               x: torch.Tensor,
               y: torch.Tensor, 
               r: torch.Tensor,
               theta: torch.Tensor,
               v: torch.Tensor,
               c: torch.Tensor,
               output_path: str,
               enable_animation: bool = True,
               animation_speed: float = 1.0,
               title: str = "SVGSplat Result"):
        """
        Export to interactive HTML with optional animation
        """
        
        # Convert tensors to numpy
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        c_np = torch.sigmoid(c).detach().cpu().numpy()
        alpha_vals = self.alpha_upper_bound * (1 / (1 + np.exp(-v_np)))
        
        # Prepare primitive data for HTML
        primitives_data = self._prepare_primitives_data(
            x_np, y_np, r_np, theta_np, c_np, alpha_vals
        )
        
        # Generate HTML
        html_content = self._generate_html(
            primitives_data, enable_animation, animation_speed, title
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Exported interactive HTML to {output_path}")
    
    def _prepare_primitives_data(self, x_np, y_np, r_np, theta_np, c_np, alpha_vals) -> List[Dict]:
        """Prepare primitive data for HTML rendering"""
        primitives = []
        N = len(x_np)
        num_primitive_types = len(self.primitive_loader.primitive_paths)
        
        for i in range(N):
            primitive_idx = i % num_primitive_types
            
            # Get embedding data for this primitive type
            embed_data = self.primitive_loader.get_html_embedding_data(primitive_idx)
            
            # Create primitive instance data
            primitive = {
                'id': i,
                'type': embed_data['type'],
                'x': float(x_np[i]),
                'y': float(y_np[i]),
                'scale': float(r_np[i]),
                'rotation': float(np.degrees(theta_np[i])),
                'color': {
                    'r': int(c_np[i][0] * 255),
                    'g': int(c_np[i][1] * 255),
                    'b': int(c_np[i][2] * 255)
                },
                'opacity': float(alpha_vals[i]),
                'width': embed_data['width'],
                'height': embed_data['height']
            }
            
            if embed_data['type'] == 'svg':
                primitive['svg_content'] = embed_data['content']
            else:
                primitive['data_url'] = embed_data['data_url']
            
            primitives.append(primitive)
        
        return primitives
    
    def _generate_html(self, primitives_data: List[Dict], enable_animation: bool, 
                      animation_speed: float, title: str) -> str:
        """Generate static HTML document with high-resolution SVG"""
        
        # Generate static SVG content directly (no JavaScript needed)
        svg_content = self._generate_static_svg(primitives_data)
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .svg-canvas {{
            border: 1px solid #ddd;
            background: white;
            display: block;
        }}
        
        .info {{
            text-align: center;
            color: #666;
            margin-top: 15px;
        }}
        
        .info h2 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 10px;
            font-size: 14px;
        }}
        
        .stat {{
            background: #f8f9fa;
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        {svg_content}
    </div>
    
    <div class="info">
        <h2>{title}</h2>
        <div class="stats">
            <div class="stat">Primitives: {len(primitives_data)}</div>
            <div class="stat">Canvas: {self.canvas_w}×{self.canvas_h}</div>
            <div class="stat">Format: Mixed SVG + Raster</div>
        </div>
        <p>High-resolution static export from SVGSplat</p>
    </div>
</body>
</html>"""
        
        return html_template
    
    def _generate_static_svg(self, primitives_data: List[Dict]) -> str:
        """Generate static SVG content with all primitives rendered"""
        
        # Start SVG with definitions for filters
        svg_parts = []
        svg_parts.append(f'<svg class="svg-canvas" width="{self.canvas_w}" height="{self.canvas_h}" viewBox="0 0 {self.canvas_w} {self.canvas_h}" xmlns="http://www.w3.org/2000/svg">')
        
        # Add definitions for color filters (for raster primitives)
        svg_parts.append('<defs>')
        for primitive in primitives_data:
            if primitive['type'] == 'raster':
                filter_id = f"colorize-{primitive['id']}"
                r = primitive['color']['r'] / 255
                g = primitive['color']['g'] / 255
                b = primitive['color']['b'] / 255
                svg_parts.append(f'''
                <filter id="{filter_id}">
                    <feColorMatrix type="matrix" values="{r} 0 0 0 0 0 {g} 0 0 0 0 0 {b} 0 0 0 0 0 1 0"/>
                </filter>''')
        svg_parts.append('</defs>')
        
        # Render all primitives (back to front order)
        center_x = self.canvas_w / 2
        center_y = self.canvas_h / 2
        
        for primitive in reversed(primitives_data):
            # Calculate transform
            transform = f"translate({primitive['x'] - center_x}, {primitive['y'] - center_y}) rotate({primitive['rotation']}) scale({primitive['scale']})"
            
            # Start group with transform and opacity
            svg_parts.append(f'<g transform="{transform}" opacity="{primitive["opacity"]}">')
            
            if primitive['type'] == 'svg':
                # Handle SVG primitive
                svg_content = primitive['svg_content']
                # Extract content between <svg> tags and apply color
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(svg_content)
                    # Apply color to all path elements
                    color_rgb = f"rgb({primitive['color']['r']},{primitive['color']['g']},{primitive['color']['b']})"
                    for elem in root.iter():
                        if elem.tag.endswith('path') or elem.tag.endswith('circle') or elem.tag.endswith('rect') or elem.tag.endswith('polygon'):
                            elem.set('fill', color_rgb)
                    
                    # Add all child elements to our group
                    for child in root:
                        svg_parts.append(ET.tostring(child, encoding='unicode'))
                except ET.ParseError:
                    # Fallback: just include the content as-is
                    svg_parts.append(svg_content.replace('<svg', '<g').replace('</svg>', '</g>'))
            
            else:
                # Handle raster primitive
                filter_id = f"colorize-{primitive['id']}"
                svg_parts.append(f'''
                <image href="{primitive['data_url']}" 
                       x="{-primitive['width']/2}" y="{-primitive['height']/2}" 
                       width="{primitive['width']}" height="{primitive['height']}" 
                       filter="url(#{filter_id})"/>''')
            
            svg_parts.append('</g>')
        
        svg_parts.append('</svg>')
        return ''.join(svg_parts)
