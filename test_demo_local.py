"""
Test script for DiffBMP demo (local testing without Gradio UI)
Tests the pipeline with a simple example
"""
import os
import sys

# Force PyTorch fallback for testing
os.environ['DIFFBMP_FORCE_PYTORCH'] = '1'

from diffbmp_pipeline import process_single_image
import json

def test_pipeline():
    """Test the pipeline with default configuration"""
    
    print("="*80)
    print("Testing DiffBMP Pipeline (PyTorch Fallback Mode)")
    print("="*80)
    
    # Check if test image exists
    test_image = "images/person/marilyn.png"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure the image exists or use a different test image")
        return False
    
    # Minimal configuration for fast testing
    config = {
        "seed": 42,
        "preprocessing": {
            "final_width": 128,  # Small for fast testing
            "img_path": test_image,
            "trim": False,
            "exist_bg": True
        },
        "primitive": {
            "primitive_file": "flowers/Gerbera1.png",
            "primitive_hollow": False,
            "output_width": 128,
            "bg_threshold": 0.5,
            "radial_transparency": False,
            "resampling": "bilinear"
        },
        "initialization": {
            "initializer": "structure_aware",
            "N": 50,  # Very few primitives for fast testing
            "v_init_bias": -4.0,
            "radii_min": 4,
            "radii_max": 20,
            "debug_mode": False
        },
        "optimization": {
            "use_fp16": False,
            "num_iterations": 10,  # Very few iterations for fast testing
            "learning_rate": {
                "default": 0.1
            },
            "c_blend": 1.0,
            "alpha_upper_bound": 1.0,
            "loss_config": {
                "type": "combined",
                "components": [
                    {"name": "grayscale_l1", "weight": 1.0},
                    {"name": "mse", "weight": 0.2}
                ]
            },
            "do_decay": True,
            "do_gaussian_blur": True,
            "blur_sigma": 1.0,
            "tile_size": 32,
            "bg_color": "white"
        },
        "postprocessing": {
            "output_folder": "outputs/demo_test/",
            "compute_psnr": False,
            "linewidth": 1.0
        }
    }
    
    try:
        print("\n📝 Configuration:")
        print(f"  - Image: {test_image}")
        print(f"  - Size: {config['preprocessing']['final_width']}x{config['preprocessing']['final_width']}")
        print(f"  - Primitives: {config['initialization']['N']}")
        print(f"  - Iterations: {config['optimization']['num_iterations']}")
        print(f"  - Output: {config['postprocessing']['output_folder']}")
        print("\n🚀 Starting processing...\n")
        
        results = process_single_image(
            img_path=test_image,
            config=config,
            output_dir=config['postprocessing']['output_folder'],
            force_cpu=False,
            disable_cuda_kernel=True
        )
        
        print("\n" + "="*80)
        print("✅ Test completed successfully!")
        print("="*80)
        print(f"📊 Results:")
        print(f"  - Output: {results['output_path']}")
        if results['pdf_path']:
            print(f"  - PDF: {results['pdf_path']}")
        print(f"  - Primitives used: {results['num_primitives']}")
        
        if results['metrics']:
            print(f"  - Metrics: {results['metrics']}")
        
        print("\n💡 To view the result, open: {results['output_path']}")
        print("\n🎉 Demo pipeline is working correctly!")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ Test failed!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
