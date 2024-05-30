import numpy as np
from PIL import Image, ImageDraw
import svgwrite
from skimage import color

# Utility functions

def create_hexagon(center, size):
    """Returns the coordinates of a hexagon centered at `center` with radius `size`."""
    angles = np.linspace(0, 2 * np.pi, 7)
    return [(center[0] + size * np.cos(angle), center[1] + size * np.sin(angle)) for angle in angles]

def hexagonal_grid(image_width, image_height, hex_size):
    """Generate a grid of hexagon centers covering the given image dimensions."""
    centers = []
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size
    for row in range(int(image_height // dy) + 1):
        for col in range(int(image_width // dx) + 1):
            x = col * dx
            y = row * dy
            if col % 2 == 1:
                y += dy / 2
            centers.append((x, y))
    return centers

def average_color_in_hexagon(image, hexagon):
    """Returns the average color of pixels inside a hexagon."""
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(hexagon, fill=1)
    mask = np.array(mask, dtype=bool)
    pixels = np.array(image)[mask]

    if len(pixels) == 0:
        # Return a default color (e.g., white) in case there are no pixels inside the hexagon
        return (255, 255, 255)

    r, g, b = np.mean(pixels, axis=0)
    return (int(r), int(g), int(b))

def find_best_match_color(color_lab, composite_colors_lab):
    """Find the best matching composite color set using LAB color space."""
    distances = np.linalg.norm(composite_colors_lab - color_lab, axis=1)
    return np.argmin(distances)

# Main process

def generate_hexagonal_art(input_image_path, output_svg_path, composite_colors, hex_size):
    image = Image.open(input_image_path).convert('RGB')
    image_width, image_height = image.size

    # Convert composite colors to LAB
    composite_colors_lab = np.array([color.rgb2lab([[c]])[0, 0] for c in composite_colors])

    # Create hexagonal grid
    hex_centers = hexagonal_grid(image_width, image_height, hex_size)

    # Initialize SVG
    dwg = svgwrite.Drawing(output_svg_path, (image_width, image_height))

    for center in hex_centers:
        hexagon = create_hexagon(center, hex_size)
        avg_color_rgb = average_color_in_hexagon(image, hexagon)
        avg_color_lab = color.rgb2lab([[avg_color_rgb]])[0, 0]

        # Find the best matching composite color
        best_match_idx = find_best_match_color(avg_color_lab, composite_colors_lab)
        best_match_color = composite_colors[best_match_idx]

        # Add hexagon to SVG using integer RGB values
        dwg.add(dwg.polygon(hexagon, fill=svgwrite.utils.rgb(*best_match_color)))

    dwg.save()

# Composite colors (you can customize this set)
composite_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 255, 255),# White
    (0, 0, 0)       # Black
]

# Parameters
input_image_path = 'input_image.jpg'  # Update with your actual image path
output_svg_path = 'hexagonal_art.svg'
hex_size = 20

# Generate hexagonal art
generate_hexagonal_art(input_image_path, output_svg_path, composite_colors, hex_size)