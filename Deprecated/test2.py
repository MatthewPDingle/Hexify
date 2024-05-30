from PIL import Image, ImageDraw
import numpy as np
import svgwrite
import math

# Load the input image
input_image = Image.open("input_image.jpg")
width, height = input_image.size

# Define the composite colors (in RGB format)
composite_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Define the hexagon size and the spacing
hexagon_radius = 50
hex_height = hexagon_radius * math.sqrt(3)
spacing = 10

# Create an SVG document
svg_width = int((hexagon_radius * 3/2) * math.ceil(width / hexagon_radius)) + spacing
svg_height = int((hex_height) * math.ceil(height / hexagon_radius / 1.5)) + spacing
dwg = svgwrite.Drawing('output.svg', size=(svg_width, svg_height))

# Function to calculate hexagon points based on the center
def hexagon(center_x, center_y, size):
    return [(center_x + size * math.cos(math.radians(angle)),
             center_y + size * math.sin(math.radians(angle)))
            for angle in range(0, 360, 60)]

# Process each hexagonal tile
for y in range(0, height, int(hex_height)):
    for x in range(0, width, 3 * hexagon_radius // 2):
        # Stagger the rows
        if (y // int(hex_height)) % 2 == 1:
            center_x = x + hexagon_radius
        else:
            center_x = x

        center_y = y + hexagon_radius

        # Ensure we don't go out of the image bounds
        if center_x >= width or center_y >= height:
            continue

        # Get the average color of the pixels in the current hexagon
        region = input_image.crop((max(0, center_x - hexagon_radius), 
                                   max(0, center_y - int(hex_height / 2)),
                                   min(width, center_x + hexagon_radius),
                                   min(height, center_y + int(hex_height / 2))))
        
        if region.size[0] > 0 and region.size[1] > 0:
            avg_color = np.array(region).mean(axis=(0, 1)).astype(int)
        else:
            avg_color = np.array([255, 255, 255], dtype=int)

        # Find the closest composite color
        closest_color = min(composite_colors, key=lambda c: sum((c - avg_color) ** 2))

        # Create a hexagonal tile with the closest composite color
        hexagon_points = hexagon(center_x, center_y, hexagon_radius)
        hexagon_shape = dwg.polygon(
            points=hexagon_points,
            fill='rgb({},{},{})'.format(*closest_color),
            stroke='black',
            stroke_width=2
        )
        dwg.add(hexagon_shape)

# Save the SVG file
dwg.save()