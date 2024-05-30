import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans
from skimage.color import rgb2lab

# Load input image
input_image = cv2.imread('input_image.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Quadruple the resolution of the output image
output_image_shape = (input_image.shape[0] * 8, input_image.shape[1] * 8, 3)
output_image = np.zeros(output_image_shape, dtype=np.uint8)

# Hexagon parameters (keeping the original size)
hex_radius = 64  # Original radius
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

# Function to draw a 12-segmented pattern
def draw_12_segment_pattern(pattern, base_radius, even_color, odd_color):
    num_segments = 12
    base_radius = int(base_radius)
    angle_step = 2 * np.pi / num_segments
    angles = np.arange(0, 2 * np.pi, angle_step)

    path_coords = []
    for angle in angles:
        x = base_radius + base_radius * np.cos(angle)
        y = base_radius + base_radius * np.sin(angle)
        path_coords.append((int(x), int(y)))

    for i in range(num_segments):
        start_point = path_coords[i]
        end_point = path_coords[(i + 1) % num_segments]

        mask = np.zeros_like(pattern[:, :, 0], dtype=np.uint8)
        print("Mask shape:", mask.shape)
        print("Point 1 (base_radius):", (base_radius, base_radius))
        print("Point 2 (start_point):", start_point)
        print("Mask dtype:", mask.dtype)
        cv2.line(mask, (base_radius, base_radius), start_point, 255, thickness=1)
        cv2.line(mask, (base_radius, base_radius), end_point, 255, thickness=1)
        cv2.fillPoly(mask, [np.array([start_point, end_point, (base_radius, base_radius)])], 255)

        color = even_color if i % 2 == 0 else odd_color
        pattern[mask != 0] = color

# Function to create concentric hexagons with patterns
def create_concentric_hex_patterns(center_x, center_y, radius, outer_black):
    pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)

    # Define sizes of each layer
    sizes = [1, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1]
    sizes = [radius * s for s in sizes]

    # Determine colors for black/white layers
    black_white_colors = [(0, 0, 0), (255, 255, 255)][::2]

    # Draw the inner layers
    for i, size in enumerate(sizes[::-1]):
        if i % 2 == 0:
            # Black or White layers (1, 3, 5, 7)
            color = outer_black if i // 2 % 2 == 0 else (255, 255, 255)
            hexagon = RegularPolygon((radius, radius), numVertices=6, radius=size, orientation=np.pi / 2)
            coords = hexagon.get_verts().astype(int)
            pattern = cv2.fillPoly(pattern, [coords], color)
        else:
            # Colored layers (2, 4, 6)
            if i == 1:  # Layer 2
                even_color, odd_color = palette[0], palette[1]
            elif i == 3:  # Layer 4
                even_color, odd_color = palette[2], palette[3]
            else:  # Layer 6
                even_color, odd_color = palette[4], palette[5]

            draw_12_segment_pattern(pattern, size, even_color, odd_color)

    return pattern

# Generate a constrained color palette using K-means clustering
num_palette_colors = 32  # Increased number of palette colors
pixels = input_image.reshape(-1, 3)
kmeans = KMeans(n_clusters=num_palette_colors, random_state=42).fit(pixels)
palette = kmeans.cluster_centers_

# Convert palette to LAB for color matching
palette_lab = rgb2lab(palette.reshape(-1, 1, 3)).reshape(-1, 3)

# Function to find the best matching palette colors using LAB color space
def best_match_colors(lab_color, palette_lab, num_matches=6):
    distances = np.linalg.norm(palette_lab - lab_color, axis=1)
    indices = np.argpartition(distances, num_matches)[:num_matches]
    return indices

# Create the hexagonal pattern grid
for center_x, center_y in hex_centers:
    if 0 <= center_x - hex_radius < output_image.shape[1] and 0 <= center_y - hex_radius < output_image.shape[0]:
        # Scale the input image coordinates to the output image coordinates
        mask = create_hex_mask(center_x, center_y, hex_radius, output_image.shape[:2])
        scaled_center_x = int(center_x / 4)
        scaled_center_y = int(center_y / 4)
        scaled_radius = hex_radius

        input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
        avg_rgb = average_color(input_image, input_mask)
        avg_lab = rgb2lab(np.array([[avg_rgb]]))[0][0]
        color_indices = best_match_colors(avg_lab, palette_lab)

        # Create a pattern using the palette colors
        outer_black = (0, 0, 0) if np.random.random() < 0.5 else (255, 255, 255)
        hex_pattern = create_concentric_hex_patterns(hex_radius, hex_radius, hex_radius, outer_black)

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