"""
core package: Core rendering and initialization modules
"""

# Renderer exports
from pydiffbmp.core.renderer.simple_tile_renderer import SimpleTileRenderer
from pydiffbmp.core.renderer.vector_renderer import VectorRenderer
from pydiffbmp.core.renderer.sequential_renderer import SequentialFrameRenderer

# Initializer exports
from pydiffbmp.core.initializer.svgsplat_initializater import StructureAwareInitializer
from pydiffbmp.core.initializer.random_initializater import RandomInitializer
from pydiffbmp.core.initializer.base_initializer import BaseInitializer

# Preprocessing
from pydiffbmp.core.preprocessing import Preprocessor

__all__ = [
    # Renderers
    'SimpleTileRenderer',
    'VectorRenderer',
    'SequentialFrameRenderer',
    
    # Initializers
    'StructureAwareInitializer',
    'RandomInitializer',
    'BaseInitializer',
    
    # Preprocessing
    'Preprocessor',
]
