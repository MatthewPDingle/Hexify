"""Create test fixture images for regression testing."""
import numpy as np
import cv2
import os

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__)) + '/fixtures'

def create_gradient_image(width=64, height=64):
    """Create a horizontal gradient from black to white."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        value = int(255 * x / (width - 1))
        img[:, x] = [value, value, value]
    return img

def create_color_blocks(width=64, height=64):
    """Create a 2x2 grid of color blocks: red, green, blue, yellow."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    h2, w2 = height // 2, width // 2
    img[0:h2, 0:w2] = [255, 0, 0]      # Red (RGB)
    img[0:h2, w2:] = [0, 255, 0]       # Green
    img[h2:, 0:w2] = [0, 0, 255]       # Blue
    img[h2:, w2:] = [255, 255, 0]      # Yellow
    return img

def create_checkerboard(width=64, height=64, cell_size=8):
    """Create a checkerboard pattern."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // cell_size) + (y // cell_size)) % 2 == 0:
                img[y, x] = [255, 255, 255]
    return img

def create_solid_gray(width=64, height=64, value=128):
    """Create a solid gray image."""
    img = np.full((height, width, 3), value, dtype=np.uint8)
    return img

def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    fixtures = {
        'gradient_64x64.png': create_gradient_image(64, 64),
        'color_blocks_64x64.png': create_color_blocks(64, 64),
        'checkerboard_64x64.png': create_checkerboard(64, 64),
        'solid_gray_64x64.png': create_solid_gray(64, 64, 128),
        'gradient_32x48.png': create_gradient_image(32, 48),  # Non-square, test aspect ratio
    }

    for name, img in fixtures.items():
        # Convert RGB to BGR for cv2.imwrite
        path = os.path.join(FIXTURES_DIR, name)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Created: {path}")

if __name__ == '__main__':
    main()
