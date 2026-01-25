"""Generate reference outputs from the current v1 code for regression testing."""
import sys
import os
import shutil

# Add parent directory to path to import hexify modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexagon_processor import HexagonProcessor
import cv2
import numpy as np

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__)) + '/fixtures'
REFERENCE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/reference_outputs'

def generate_reference(input_path, output_name, num_colors=16):
    """Generate a reference output for a test fixture."""
    print(f"Processing: {input_path} -> {output_name}")

    input_image = cv2.imread(input_path)
    if input_image is None:
        print(f"  ERROR: Could not read {input_path}")
        return False

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    processor = HexagonProcessor(
        num_palette_colors=num_colors,
        num_processes=1,  # Single process for determinism
        hexagons_dir=None,
        chunk_size=32,
        save_hexagons=False
    )

    output_image = processor.process_image(input_image)

    # Save output
    output_path = os.path.join(REFERENCE_DIR, output_name)
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output_path}")

    # Also save the palette for reference
    palette_path = os.path.join(REFERENCE_DIR, output_name.replace('.png', '_palette.npy'))
    np.save(palette_path, processor.palette)
    print(f"  Palette: {palette_path}")

    return True

def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    test_cases = [
        ('gradient_64x64.png', 'ref_gradient_64x64_c16.png', 16),
        ('color_blocks_64x64.png', 'ref_color_blocks_64x64_c16.png', 16),
        # NOTE: checkerboard and solid_gray crash due to bug in original code with low-color images
        # ('checkerboard_64x64.png', 'ref_checkerboard_64x64_c5.png', 5),
        # ('solid_gray_64x64.png', 'ref_solid_gray_64x64_c5.png', 5),
        ('gradient_32x48.png', 'ref_gradient_32x48_c16.png', 16),
        ('gradient_64x64.png', 'ref_gradient_64x64_c8.png', 8),  # Test different palette size
    ]

    # Also test with a real image if fire.png exists
    fire_path = os.path.join(os.path.dirname(FIXTURES_DIR), '..', 'fire.png')
    if os.path.exists(fire_path):
        # Copy fire.png to fixtures and create a small version
        import cv2
        fire_img = cv2.imread(fire_path)
        if fire_img is not None:
            # Create a small version for faster testing
            small_fire = cv2.resize(fire_img, (64, 64))
            small_fire_path = os.path.join(FIXTURES_DIR, 'fire_64x64.png')
            cv2.imwrite(small_fire_path, small_fire)
            test_cases.append(('fire_64x64.png', 'ref_fire_64x64_c16.png', 16))

    for fixture_name, output_name, num_colors in test_cases:
        input_path = os.path.join(FIXTURES_DIR, fixture_name)
        if os.path.exists(input_path):
            generate_reference(input_path, output_name, num_colors)
        else:
            print(f"SKIP: {input_path} not found")

if __name__ == '__main__':
    main()
