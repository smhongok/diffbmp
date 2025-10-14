"""
setup.py for image-as-brush
Make this package pip installable with: pip install -e .

Note: This setup.py does NOT install dependencies.
Assumes the environment is already properly configured.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='image-as-brush',  # pip install name (can use dash)
    version='0.1.0',
    description='Fast Differentiable Painting with Any Image Primitives',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='anonymous',
    author_email='',
    url='https://github.com/your-repo/image-as-brush',
    
    # Package discovery: Python import name is 'imbrush'
    packages=['imbrush', 'imbrush.core', 'imbrush.core.renderer', 
              'imbrush.core.initializer', 'imbrush.util'],
    
    # No dependencies - assumes environment is already set up
    # Users should manually install requirements if needed
    install_requires=[],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Additional metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    # Keywords
    keywords='differentiable-rendering 2d-splatting image-synthesis pytorch cuda',
    
    # Include package data
    include_package_data=True,
    
    # Entry points (optional - can add CLI tools later)
    entry_points={
        'console_scripts': [
            # 'image-as-brush=image_as_brush.cli:main',  # Future CLI tool
        ],
    },
    
    # Extras for development
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
)
