import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

# Load input image
input_image = cv2.imread('input_image.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Double the resolution of the output image
output_image_shape = (input_image.shape[0] * 2, input_image.shape[1] * 2, 3)
output_image = np.zeros(output_image_shape, dtype=np.uint8)

# Hexagon parameters (keeping the original size)
hex_radius = 30  # Original radius
hex_width = np.sqrt(3) * hex_radius
hex_height = 2 * hex_radius
cols = int(output_image.shape[1] // hex_width) + 1
rows = int(output_image.shape[0] // (hex_height * 0.75)) + 1

# Create a list of hexagon centers in a staggered grid
hex_centers = [
    (
        int(hex_width * col),
        int(hex_height * 1 * row + (0.5 * hex_height if col % 2 else 0))
    )
    for row in range(rows)
    for col in range(cols)
]

# Function to calculate the average color of a region
def average_color(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    avg_color = masked[np.where(mask != 0)].mean(axis=0)
    return avg_color

# Function to create a hexagon mask
def create_hex_mask(center_x, center_y, radius, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts()
    coords = np.clip(coords, [0, 0], [shape[1] - 1, shape[0] - 1]).astype(int)
    mask = cv2.fillPoly(mask, [coords], 255)
    return mask

# Function to create concentric hexagons with patterns
def create_hex_pattern(center_x, center_y, radius, colors):
    pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
    hexagon = RegularPolygon((radius, radius), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts().astype(int)

    for i, color in enumerate(colors):
        hex_radius = radius * (1 - 0.2 * i)
        inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
        inner_coords = inner_hexagon.get_verts().astype(int)

        pattern = cv2.fillPoly(pattern, [inner_coords], color)

    return pattern

# Generate a constrained color palette using K-means clustering
num_palette_colors = 8
pixels = input_image.reshape(-1, 3)
kmeans = KMeans(n_clusters=num_palette_colors, random_state=42).fit(pixels)
palette = kmeans.cluster_centers_

# Convert palette to LAB for color matching
palette_lab = rgb2lab(palette.reshape(-1, 1, 3)).reshape(-1, 3)

# Function to find the best matching palette colors using LAB color space
def best_match_colors(lab_color, palette_lab, num_matches=3):
    distances = np.linalg.norm(palette_lab - lab_color, axis=1)
    indices = np.argpartition(distances, num_matches)[:num_matches]
    return indices

# Create the hexagonal pattern grid
for center_x, center_y in hex_centers:
    if 0 <= center_x - hex_radius < output_image.shape[1] and 0 <= center_y - hex_radius < output_image.shape[0]:
        # Scale the input image coordinates to the output image coordinates
        mask = create_hex_mask(center_x, center_y, hex_radius, output_image.shape[:2])
        scaled_center_x = int(center_x / 2)
        scaled_center_y = int(center_y / 2)
        scaled_radius = hex_radius

        input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
        avg_rgb = average_color(input_image, input_mask)
        avg_lab = rgb2lab(np.array([[avg_rgb]]))[0][0]
        color_indices = best_match_colors(avg_lab, palette_lab)

        # Create a pattern using the palette colors
        pattern_colors = [palette[i] for i in color_indices]
        hex_pattern = create_hex_pattern(hex_radius, hex_radius, hex_radius, pattern_colors)

        # Mask the pattern to the hex shape
        mask_pattern = create_hex_mask(hex_radius, hex_radius, hex_radius, hex_pattern.shape[:2])
        hex_pattern_masked = cv2.bitwise_and(hex_pattern, hex_pattern, mask=mask_pattern)

        # Blend the pattern into the output image
        x_start = max(center_x - hex_radius, 0)
        y_start = max(center_y - hex_radius, 0)
        x_end = min(center_x + hex_radius, output_image.shape[1])
        y_end = min(center_y + hex_radius, output_image.shape[0])

        hex_slice = output_image[y_start:y_end, x_start:x_end]
        pattern_slice = hex_pattern_masked[:y_end - y_start, :x_end - x_start]
        mask_slice = mask[y_start:y_end, x_start:x_end]

        hex_slice[mask_slice != 0] = pattern_slice[mask_slice != 0]

# Save and display the output image
plt.imsave('output.png', output_image)
plt.figure(figsize=(15, 15))
plt.imshow(output_image)
plt.axis('off')
plt.show()