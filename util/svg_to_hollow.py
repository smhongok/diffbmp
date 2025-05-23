#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from xml.dom import minidom
from xml.etree import ElementTree as ET

def convert_svg_to_hollow(input_file, output_file, stroke_width=None, stroke_color='#000000'):
    """
    Function to convert SVG file to hollow format
    
    Args:
        input_file (str): Input SVG file path
        output_file (str): Output SVG file path
        stroke_width (float): Border thickness (None for automatic calculation)
        stroke_color (str): Border color
    """
    # Register XML namespace
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    
    try:
        # Parse SVG file
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # Define SVG namespace
        svg_ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Get original viewBox and size information
        viewBox = root.get('viewBox')
        width = root.get('width')
        height = root.get('height')
        
        # Adjust border thickness based on image size
        adjusted_width = None
        
        # If viewBox doesn't exist, create from width and height
        if not viewBox and width and height:
            try:
                w = float(width.rstrip('px').rstrip('em').rstrip('pt').rstrip('%'))
                h = float(height.rstrip('px').rstrip('em').rstrip('pt').rstrip('%'))
                viewBox = f"0 0 {w} {h}"
                root.set('viewBox', viewBox)
                print(f"Created viewBox because it didn't exist: {viewBox}")
            except (ValueError, TypeError):
                pass
        
        # Automatic border thickness calculation
        if viewBox:
            # Calculate appropriate border thickness proportional to viewBox size
            vb_parts = viewBox.split()
            if len(vb_parts) == 4:
                vb_min_x = float(vb_parts[0])
                vb_min_y = float(vb_parts[1])
                vb_width = float(vb_parts[2])
                vb_height = float(vb_parts[3])
                
                # Automatic border thickness calculation based on image size
                if stroke_width is None:
                    # Small image
                    if min(vb_width, vb_height) <= 50:
                        adjusted_width = min(vb_width, vb_height) * 0.02  # 2%
                    # Medium size image
                    elif min(vb_width, vb_height) <= 200:
                        adjusted_width = min(vb_width, vb_height) * 0.01  # 1%
                    # Large image
                    else:
                        adjusted_width = min(vb_width, vb_height) * 0.005  # 0.5%
                    
                    # Set min/max value range
                    adjusted_width = max(adjusted_width, 0.5)  # Minimum 0.5
                    print(f"Border thickness automatically calculated: {adjusted_width:.1f}")
                else:
                    # Use user specified value
                    adjusted_width = stroke_width
                
                # Expand viewBox to account for border thickness
                # Expand by half the border thickness in each direction (border grows from center)
                padding = adjusted_width / 2
                
                # Set new viewBox (add padding to prevent border from being cut off)
                new_viewBox = f"{vb_min_x - padding} {vb_min_y - padding} {vb_width + padding*2} {vb_height + padding*2}"
                root.set('viewBox', new_viewBox)
        
        # Set default border thickness if viewBox doesn't exist
        if adjusted_width is None:
            if stroke_width is not None:
                adjusted_width = stroke_width
            else:
                adjusted_width = 1.0  # Default value
        
        # Adjust other attributes to account for border thickness
        if width and height:
            try:
                # Extract units
                w_str = width
                h_str = height
                
                w_unit = ""
                h_unit = ""
                
                # Separate numeric part and unit part
                w_match = re.match(r'([0-9.]+)([a-z%]*)', w_str)
                h_match = re.match(r'([0-9.]+)([a-z%]*)', h_str)
                
                if w_match:
                    w = float(w_match.group(1))
                    w_unit = w_match.group(2)
                else:
                    w = float(w_str)
                
                if h_match:
                    h = float(h_match.group(1))
                    h_unit = h_match.group(2)
                else:
                    h = float(h_str)
                
                # Calculate expanded size (padding is border thickness)
                padding = adjusted_width
                w_new = w + padding * 2
                h_new = h + padding * 2
                
                # Set new size
                root.set('width', f"{w_new}{w_unit}")
                root.set('height', f"{h_new}{h_unit}")
            except (ValueError, TypeError) as e:
                print(f"Error during size adjustment: {e}")
            
        # Find or create <style> element
        style_elem = root.find('.//svg:style', svg_ns)
        
        if style_elem is None:
            # Create style element if it doesn't exist
            style_elem = ET.SubElement(root, '{http://www.w3.org/2000/svg}style')
            style_elem.set('type', 'text/css')
            style_elem.text = '\n\t.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}\n'
        else:
            # Update or add hollow style to existing style element
            style_text = style_elem.text or ""
            if '.st0' in style_text:
                # Update existing st0 class
                style_text = re.sub(r'\.st0\{[^}]*\}', '.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + 
                                   str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}', style_text)
                style_elem.text = style_text
            else:
                # Add new st0 class
                style_elem.text = style_text + '\n\t.st0{fill:none;stroke:' + stroke_color + ';stroke-width:' + str(adjusted_width) + ';stroke-linejoin:round;stroke-linecap:round;}\n'
        
        # Apply hollow style to all path, rect, circle, ellipse and other elements
        for elem in root.findall('.//{http://www.w3.org/2000/svg}path') + \
                    root.findall('.//{http://www.w3.org/2000/svg}rect') + \
                    root.findall('.//{http://www.w3.org/2000/svg}circle') + \
                    root.findall('.//{http://www.w3.org/2000/svg}ellipse') + \
                    root.findall('.//{http://www.w3.org/2000/svg}polygon'):
            
            # Save original style information
            original_style = {}
            if 'style' in elem.attrib:
                style_text = elem.attrib['style']
                # Preserve important original style attributes
                stroke_match = re.search(r'stroke-width:([^;]+);', style_text)
                if stroke_match:
                    original_style['stroke-width'] = stroke_match.group(1)
                
                # Remove only fill attribute from existing style
                style_text = re.sub(r'fill:[^;]+;', '', style_text)
                elem.attrib['style'] = style_text
            
            # Remove fill attribute
            if 'fill' in elem.attrib:
                del elem.attrib['fill']
                
            # Set class attribute while maintaining original border attributes
            elem.set('class', 'st0')
            
            # Set user-specified border thickness
            if 'stroke-width' in original_style:
                elem.set('stroke-width', original_style['stroke-width'])
            else:
                elem.set('stroke-width', str(adjusted_width))
            
            # Set border color
            elem.set('stroke', stroke_color)
            elem.set('fill', 'none')
            
        # Convert result to string
        rough_string = ET.tostring(root, 'utf-8')
        
        # Use minidom for cleaner XML formatting
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent='\t')
        
        # Remove empty lines
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])
        
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        print(f"Hollow SVG successfully generated: {output_file}")
        print(f"Applied border thickness: {adjusted_width}")
        if viewBox:
            print(f"Original viewBox: {viewBox}")
            print(f"Modified viewBox: {new_viewBox if 'new_viewBox' in locals() else 'No change'}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function: Process command line arguments and call SVG conversion function
    """
    parser = argparse.ArgumentParser(description='Convert SVG file to hollow format')
    parser.add_argument('input_file', help='Input SVG file path')
    parser.add_argument('--output', '-o', help='Output SVG file path (default: append _hollow to input_file)')
    parser.add_argument('--stroke-width', '-w', type=float, 
                        help='Border thickness (default: automatic calculation based on image size)')
    parser.add_argument('--stroke-color', '-c', default='#000000', help='Border color (default: #000000)')
    
    args = parser.parse_args()
    
    # If output file name is not specified, append _hollow to input file name
    if args.output is None:
        base_name, ext = os.path.splitext(args.input_file)
        args.output = f"{base_name}_hollow{ext}"
    
    convert_svg_to_hollow(args.input_file, args.output, args.stroke_width, args.stroke_color)

if __name__ == '__main__':
    main() 