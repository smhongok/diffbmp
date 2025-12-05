"""
diffbmp: Differentiable Rendering with Bitmap Primitives

A differentiable rendering framework for image synthesis using arbitrary primitives
(SVG shapes, text glyphs, raster images) as "brushes" in a 2D splatting pipeline.

Key Features:
- GPU-accelerated tile-based rendering with CUDA
- Differentiable end-to-end pipeline
- Support for SVG, text, and raster primitives
- PSD/PDF export for designer workflows
- Structure-aware initialization

Example:
    >>> import pydiffbmp
    >>> from pydiffbmp import SimpleTileRenderer, PrimitiveLoader
    >>> 
    >>> # Load primitive
    >>> loader = PrimitiveLoader("circle.svg", output_width=256, device='cuda')
    >>> S = loader.load_alpha_bitmap()
    >>> 
    >>> # Initialize renderer
    >>> renderer = SimpleTileRenderer(
    ...     canvas_size=(512, 512),
    ...     S=S,
    ...     device='cuda'
    ... )
    >>> 
    >>> # Render from parameters
    >>> output = renderer.render_from_params(x, y, r, theta, v, c)
"""

__version__ = '0.1.0'
__author__ = 'Anonymous'

# ============================================================================
# Functional API (PyTorch-style) - Recommended for most users
# ============================================================================
from pydiffbmp.functional import (
    load_primitive,
    initialize_params,
    render,
    preprocess_image,
)

# ============================================================================
# Class-based API - For advanced users who need more control
# ============================================================================

# Core renderer exports
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
from pydiffbmp.core.renderer.sequential_renderer import SequentialFrameRenderer

# Initializer exports
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.initializer.base_initializer import BaseInitializer

# Utility exports
from pydiffbmp.util.primitive_loader import PrimitiveLoader
from pydiffbmp.util.svg_loader import SVGLoader
from pydiffbmp.util.psd_exporter import PSDExporter
from pydiffbmp.util.pdf_exporter import PDFExporter
from pydiffbmp.util.loss_functions import LossComposer

# Preprocessing
from pydiffbmp.core.preprocessing import Preprocessor

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Functional API (Recommended)
    'load_primitive',
    'initialize_params',
    'render',
    'preprocess_image',
    
    # Renderers
    'SimpleTileRenderer',
    'VectorRenderer',
    'SequentialFrameRenderer',
    
    # Initializers
    'StructureAwareInitializer',
    'RandomInitializer',
    'BaseInitializer',
    
    # Loaders
    'PrimitiveLoader',
    'SVGLoader',
    
    # Exporters
    'PSDExporter',
    'PDFExporter',
    
    # Utilities
    'LossComposer',
    'Preprocessor',
]


def get_version():
    """Return the current version of diffbmp."""
    return __version__


def check_cuda_available():
    """Check if CUDA extensions are properly compiled and available."""
    try:
        import sys
        import os
        cuda_path = os.path.join(os.path.dirname(__file__), 'cuda_tile_rasterizer', 'cuda_tile_rasterizer')
        if cuda_path not in sys.path:
            sys.path.insert(0, cuda_path)
        from cuda_tile_rasterizer import TileRasterizer
        return True
    except ImportError:
        return False


# Print helpful message on import
def _print_welcome():
    import sys
    if 'pytest' not in sys.modules:  # Don't print during tests
        cuda_status = "✓ Available" if check_cuda_available() else "✗ Not compiled"
        print(f"diffbmp v{__version__} loaded")
        print(f"CUDA extensions: {cuda_status}")
        if not check_cuda_available():
            print("  → Build CUDA extensions: cd cuda_tile_rasterizer && python setup.py build_ext --inplace")

# Uncomment to show welcome message on import
# _print_welcome()
