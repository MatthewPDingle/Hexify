import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

# Load input image
input_image = cv2.imread('test.jpg')
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Quadruple the resolution of the output image
output_image_shape = (input_image.shape[0] * 16, input_image.shape[1] * 16, 3)
output_image = np.zeros(output_image_shape, dtype=np.uint8)

# Hexagon parameters with a diameter of 256 pixels
hex_diameter = 256
hex_radius = hex_diameter // 2
hex_width = 2 * hex_radius
hex_height = math.sqrt((hex_radius * hex_radius) - (hex_radius / 2 * hex_radius / 2)) * 2
cols = int(output_image.shape[1] // (hex_width - (hex_width / 4))) + 1
rows = int(output_image.shape[0] // (hex_height * 0.75)) + 1

# Create a list of hexagon centers in a staggered grid
hex_centers = [
    (
        int((hex_width - 64) * col),
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

# Function to determine if a color is closer to black or white in LAB space
def black_or_white(lab_color):
    L, a, b = lab_color
    return [0, 0, 0] if L < 50 else [255, 255, 255]

# Function to calculate the area of a sector of a hexagon
def hex_sector_area(radius, angle_deg):
    angle_rad = math.radians(angle_deg)
    return 0.5 * radius * radius * angle_rad

# Function to mix two LAB colors with a given weight
def mix_colors(lab_color1, lab_color2, weight):
    """Mix two LAB colors with a given weight."""
    mixed_lab = (1 - weight) * lab_color1 + weight * lab_color2
    return mixed_lab

# Function to clamp LAB values to their valid ranges
def clamp_lab(lab_color):
    L = np.clip(lab_color[0], 0, 100)
    a = np.clip(lab_color[1], -128, 127)
    b = np.clip(lab_color[2], -128, 127)
    return np.array([L, a, b])

# Function to find the closest pair of colors (one from vivid, one from dark) that mix to approximate the avg_lab
def closest_palette_colors(avg_lab, vivid_palette_lab, dark_palette_lab, even_area, odd_area):
    min_distance = float('inf')
    best_pair = (None, None)
    best_weight = even_area / (even_area + odd_area)
    
    for vivid_color in vivid_palette_lab:
        for dark_color in dark_palette_lab:
            mixed_lab = mix_colors(vivid_color, dark_color, best_weight)
            mixed_lab = clamp_lab(mixed_lab)  # Ensure LAB values are clamped
            distance = np.linalg.norm(avg_lab - mixed_lab)
            
            if distance < min_distance:
                min_distance = distance
                best_pair = (vivid_color, dark_color)
    
    return best_pair, best_weight

# Function to create concentric hexagons with patterns
def create_hex_pattern(center_x, center_y, radius, avg_lab, vivid_palette, vivid_palette_lab, dark_palette, dark_palette_lab):
    pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
    hexagon = RegularPolygon((radius, radius), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts().astype(int)
    
    for i in range(7, 0, -1):  # Iterate from outer to inner layer
        if i == 7:  # Outer layer (layer 7)
            hex_radius = radius * (i / 7)
            hex_radius = int(hex_radius)
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)
        
            color = black_or_white(avg_lab)
            pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, color)))
        elif i == 6:  # Layer 6
            L, a, b = avg_lab
            diameter = 256 - abs(L - 50) * (256 - 192) / 50 
            hex_radius = int(diameter // 2)
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            # Randomly determine the angles
            even_angle = np.random.randint(1, 60)
            odd_angle = 60 - even_angle

            # Calculate areas of even and odd-numbered zones
            even_area = 6 * hex_sector_area(hex_radius, even_angle)
            odd_area = 6 * hex_sector_area(hex_radius, odd_angle)

            # Find the best matching colors from the palettes
            (vivid_color, dark_color), best_weight = closest_palette_colors(avg_lab, vivid_palette_lab, dark_palette_lab, even_area, odd_area)
            
            # Convert LAB colors to RGB for filling polygons
            vivid_color_rgb = (lab2rgb(clamp_lab(vivid_color).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]
            dark_color_rgb = (lab2rgb(clamp_lab(dark_color).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]
            
            # Ensure RGB values are tuples of integers
            vivid_color_rgb = tuple(map(int, vivid_color_rgb))
            dark_color_rgb = tuple(map(int, dark_color_rgb))
            
            # Draw the 12 zones with alternating colors
            angle_offset = 30 - (even_angle / 2)  # Start from half the interior angle of the first zone
            for zone in range(12):
                if zone % 2 == 0:
                    angle = even_angle
                    angle_color = vivid_color_rgb
                else:
                    angle = odd_angle
                    angle_color = dark_color_rgb
                
                start_angle = angle_offset
                end_angle = start_angle + angle
                angle_offset += angle

                # Calculate the vertices of p1, p2, and p3.
                p1 = (radius + hex_radius * math.cos(math.radians(start_angle)), radius + hex_radius * math.sin(math.radians(start_angle)))
                p2 = (radius + hex_radius * math.cos(math.radians(end_angle)), radius + hex_radius * math.sin(math.radians(end_angle)))
                
                # Calculate the midpoint p12
                p12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

                # Calculate the distance between p1 and p12 (half the hypotenuse)
                d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 / 2

                # Length of the shorter leg in 30-60-90 triangle
                length_of_shorter_leg = d / math.sqrt(3)

                # Direction vector from p1 to p12
                dx = p12[0] - p1[0]
                dy = p12[1] - p1[1]

                # Normalize the direction vector
                length = (dx ** 2 + dy ** 2) ** 0.5
                dx /= length
                dy /= length

                # Coordinates of p3
                p3 = (p12[0] + length_of_shorter_leg * dy, p12[1] - length_of_shorter_leg * dx)
                
                # Clip the points to stay within inner_coords
                def clip_point_to_hexagon(point, hex_coords):
                    path = Path(hex_coords)
                    if path.contains_point(point):
                        return point

                    # Find the nearest point on the hexagon edges
                    min_dist = float('inf')
                    nearest_point = point
                    for i in range(len(hex_coords)):
                        p1 = hex_coords[i]
                        p2 = hex_coords[(i + 1) % len(hex_coords)]
                        nearest = nearest_point_on_segment(point, p1, p2)
                        dist = np.linalg.norm(np.array(nearest) - np.array(point))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_point = nearest
                    return nearest_point

                def nearest_point_on_segment(point, p1, p2):
                    px, py = point
                    x1, y1 = p1
                    x2, y2 = p2

                    dx, dy = x2 - x1, y2 - y1
                    if dx == dy == 0:  # the segment's start and end points are the same
                        return p1

                    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
                    t = max(0, min(1, t))
                    return (x1 + t * dx, y1 + t * dy)

                p1 = clip_point_to_hexagon(p1, inner_coords)
                p2 = clip_point_to_hexagon(p2, inner_coords)
                p3 = clip_point_to_hexagon(p3, inner_coords)
                
                # Fill the main part of the zone
                vertices1 = [(radius, radius), p1, p2]
                vertices1 = np.array(vertices1, dtype=np.int32)
                cv2.fillPoly(pattern, [vertices1], angle_color)
                
                # Fill the rest of the zone on the outside that requires another polygon
                if zone % 2 == 1:
                    vertices2 = [p1, p2, p3]
                    vertices2 = np.array(vertices2, dtype=np.int32)
                    cv2.fillPoly(pattern, [vertices2], angle_color)
                 
    return pattern

# Reshape the image into a 2D array of pixels
pixels = input_image.reshape(-1, 3)

# Separate image pixels into blue-based, red-based, and green-based
blue_pixels = pixels[pixels[:, 2] > np.maximum(pixels[:, 0], pixels[:, 1])]
red_pixels = pixels[pixels[:, 0] > np.maximum(pixels[:, 1], pixels[:, 2])]
green_pixels = pixels[pixels[:, 1] > np.maximum(pixels[:, 0], pixels[:, 2])]

# Apply K-means clustering separately to each group
num_palette_colors_per_group = 3

def cluster_pixels(pixels, num_clusters):
    if len(pixels) > 0:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
        return kmeans.cluster_centers_
    return np.array([])

blue_palette = cluster_pixels(blue_pixels, num_palette_colors_per_group)
red_palette = cluster_pixels(red_pixels, num_palette_colors_per_group)
green_palette = cluster_pixels(green_pixels, num_palette_colors_per_group)

# Combine the clusters to form the initial palette of 9 colors
palette = np.vstack((blue_palette, red_palette, green_palette))

# Function to make colors vivid in RGB space by setting the highest channel to 255
def make_vivid(color):
    indices = np.argsort(color)
    color[indices[2]] = 255  # Set the highest channel to 255
    return color

# Function to make colors dark in RGB space by cutting the middle channel by 75% and the highest channel by 50%
def make_dark(color):
    indices = np.argsort(color)
    color[indices[1]] = color[indices[1]] // 4  # Cut the middle channel by 75%
    color[indices[2]] = color[indices[2]] // 2  # Cut the highest channel by 50%
    return color

# Apply the vividness adjustment to the initial palette
vivid_palette = np.array([make_vivid(color) for color in palette])

# Create an additional palette of 9 dark colors
dark_palette = np.array([make_dark(color) for color in palette])

# Sort each set of 9 colors by their brightness in the LAB color space
vivid_palette_lab = rgb2lab(vivid_palette.reshape(-1, 1, 3)).reshape(-1, 3)
dark_palette_lab = rgb2lab(dark_palette.reshape(-1, 1, 3)).reshape(-1, 3)

sorted_vivid_indices = np.argsort(vivid_palette_lab[:, 0])
sorted_dark_indices = np.argsort(dark_palette_lab[:, 0])

sorted_vivid_palette = vivid_palette[sorted_vivid_indices]
sorted_dark_palette = dark_palette[sorted_dark_indices]

# Combine the sorted palettes
final_sorted_palette = np.vstack((sorted_vivid_palette, sorted_dark_palette))

# Create and save the palette image with bright colors on top and dark colors below
palette_image = np.zeros((64, 32 * 9, 3), dtype=np.uint8)  # 64 pixels high, 32 pixels wide per color

for i, color in enumerate(sorted_vivid_palette):
    palette_image[:32, i * 32:(i + 1) * 32] = color  # Top half for vivid colors

for i, color in enumerate(sorted_dark_palette):
    palette_image[32:, i * 32:(i + 1) * 32] = color  # Bottom half for dark colors

plt.imsave('palette.png', palette_image)

# Create the hexagonal pattern grid
counter = 0
for center_x, center_y in hex_centers:
    if 0 <= center_x - hex_radius < output_image.shape[1] and 0 <= center_y - hex_radius < output_image.shape[0]:
        # Scale the input image coordinates to the output image coordinates
        mask = create_hex_mask(center_x, center_y, hex_radius, output_image.shape[:2])
        scaled_center_x = int(center_x / 16)
        scaled_center_y = int(center_y / 16)
        scaled_radius = hex_radius // 16

        input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
        avg_rgb = average_color(input_image, input_mask)
        avg_lab = rgb2lab(np.array([[avg_rgb]], dtype=np.float32) / 255.0)[0][0]  # Proper scaling

        # Create a pattern using the average LAB color
        hex_pattern = create_hex_pattern(hex_radius, hex_radius, hex_radius, avg_lab, sorted_vivid_palette, vivid_palette_lab, sorted_dark_palette, dark_palette_lab)

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