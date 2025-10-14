#!/usr/bin/env python
"""Test imbrush import"""

print("Testing imbrush imports...")

# Test 1: Basic import
import imbrush
print(f"✓ import imbrush - Version: {imbrush.__version__}")

# Test 2: Top-level exports
from imbrush import SimpleTileRenderer, PrimitiveLoader
print(f"✓ from imbrush import SimpleTileRenderer, PrimitiveLoader")

# Test 3: Subpackage imports
from imbrush.core import StructureAwareInitializer
from imbrush.util import PSDExporter
print(f"✓ from imbrush.core import StructureAwareInitializer")
print(f"✓ from imbrush.util import PSDExporter")

# Test 4: Direct submodule imports
from imbrush.core.renderer import VectorRenderer
from imbrush.core.initializer import RandomInitializer
print(f"✓ from imbrush.core.renderer import VectorRenderer")
print(f"✓ from imbrush.core.initializer import RandomInitializer")

print("\n" + "="*60)
print("✓ All imports successful!")
print("="*60)
print("\nYou can now use in SVGDreamer:")
print("  from imbrush import SimpleTileRenderer, PrimitiveLoader")
print("  from imbrush.core import StructureAwareInitializer")
print("  from imbrush.util import PSDExporter")
