import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

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

def closest_palette_color(avg_lab, palette_lab, avoid_lab=None):
    min_distance = float('inf')
    closest_color = None

    for color in palette_lab:
        if avoid_lab is not None and any(np.array_equal(color, avoid_color) for avoid_color in avoid_lab):
            continue
        
        distance = np.linalg.norm(avg_lab - color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    
    return closest_color
    
def select_color_2(avg_lab, palette_lab, color_1, color_1_area, color_2_area, bw_color, bw_color_area):
    min_distance = float('inf')
    best_color_2 = None
    
    # Ensure color_1 and bw_color are NumPy arrays
    color_1 = np.array(color_1)
    bw_color = np.array(bw_color)
    
    for color in palette_lab:
        if np.array_equal(color, color_1):
            continue
        
        color = np.array(color)  # Ensure the current palette color is a NumPy array
        
        # Calculate the mixed LAB color
        mixed_lab = (color_1_area * color_1 + color_2_area * color + bw_color_area * bw_color) / (color_1_area + color_2_area + bw_color_area)
        
        # Calculate the distance to the average LAB color
        distance = np.linalg.norm(avg_lab - mixed_lab)
        
        if distance < min_distance:
            min_distance = distance
            best_color_2 = color
    
    return best_color_2

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

# Function to create concentric hexagons with patterns
def create_hex_pattern(center_x, center_y, radius, avg_lab, palette_lab):
    pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
    hexagon = RegularPolygon((radius, radius), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts().astype(int)

    layer_areas = []
    bw_color = (0, 0, 0)
    layer_7_radius = 0;
    layer_6_radius = 0;
    layer_5_radius = 0;
    layer_4_radius = 0;
    layer_3_radius = 0;
    avoid_lab = []

    for i in range(7, 0, -1):  # Iterate from outer to inner layer
        if i == 7:  # Outer layer (layer 7)
            hex_radius = radius * (i / 7)
            hex_radius = int(hex_radius)
            layer_7_radius = hex_radius
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            bw_color = (0, 0, 0) if avg_lab[0] < 50 else (255, 255, 255)
            pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
            
            # Calculate and store the area of layer 7
            layer_7_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
            layer_areas.append(layer_7_area)
            
        elif i == 6:  # Layer 6
            L, a, b = avg_lab
            # radius - .5 * width
            diameter = 256 - abs(L - 50) * (256 - 192) / 50 
            hex_radius = int(diameter // 2)
            layer_6_radius = hex_radius
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            # Select the color from the palette that most closely matches the avg_lab
            color_1 = closest_palette_color(avg_lab, palette_lab)
            avoid_lab.append(color_1)
            dist_1 = np.linalg.norm(color_1 - avg_lab)

            # Find the next color from the palette that is most similar to color_1
            color_adj = min(palette_lab, key=lambda color: np.linalg.norm(color - color_1) if not np.array_equal(color, color_1) else float('inf'))
            dist_adj = np.linalg.norm(color_1 - color_adj)

            # Determine the percentage dist_1 is of dist_adj
            percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
            even_angle = percentage_off * 60
            odd_angle = 60 - even_angle
            
            # Calculate areas of even and odd-numbered zones
            even_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
            odd_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))

            # Sum of even_area and odd_area
            layer_6_area = even_area + odd_area
            layer_areas.append(layer_6_area)

            # Select color 2 from the palette that most closely matches the avg_lab
            color_2 = select_color_2(avg_lab, palette_lab, color_1, even_area, odd_area, bw_color, layer_areas[0] - layer_6_area)

            # Convert LAB colors to RGB for filling polygons
            color_1_rgb = (lab2rgb(clamp_lab(color_1).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]
            color_2_rgb = (lab2rgb(clamp_lab(color_2).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]

            # Ensure RGB values are tuples of integers
            color_1_rgb = tuple(map(int, color_1_rgb))
            color_2_rgb = tuple(map(int, color_2_rgb))

            # Draw the 12 zones with alternating colors
            angle_offset = 30 - (even_angle / 2)  # Start from half the interior angle of the first zone
            for zone in range(12):
                if zone % 2 == 0:
                    angle = even_angle
                    angle_color = color_1_rgb
                else:
                    angle = odd_angle
                    angle_color = color_2_rgb

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

                # Clip the points to ensure they're not bigger than desired
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
                    
        elif i == 5:  # Layer 5
            hex_radius = radius * (i / 7)
            hex_radius = int(hex_radius)
            layer_5_radius = hex_radius
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
            
            # Calculate and store the area of layer 5
            layer_5_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
            layer_areas.append(layer_5_area)
            
        elif i == 4:  # Layer 4
            L, a, b = avg_lab
            diameter = (layer_5_radius * 2) - abs(L - 50) * (64) / 50 
            hex_radius = int(diameter // 2)
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            # Scale the input image coordinates to the output image coordinates
            mask = create_hex_mask(center_x, center_y, hex_radius, output_image.shape[:2])
            scaled_center_x = int(center_x / 16)
            scaled_center_y = int(center_y / 16)
            scaled_radius = hex_radius // 16

            input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
            avg_rgb = average_color(input_image, input_mask)
            avg_lab = rgb2lab(np.array([[avg_rgb]], dtype=np.float32) / 255.0)[0][0]  # Proper scaling

            # Select the color from the palette that most closely matches the avg_lab
            print(avg_lab, avoid_lab)
            color_1 = closest_palette_color(avg_lab, palette_lab, avoid_lab)
            dist_1 = np.linalg.norm(color_1 - avg_lab)

            # Find the next color from the palette that is most similar to color_1
            color_adj = min(palette_lab, key=lambda color: np.linalg.norm(color - color_1) if not np.array_equal(color, color_1) else float('inf'))
            dist_adj = np.linalg.norm(color_1 - color_adj)

            # Determine the percentage dist_1 is of dist_adj
            percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
            even_angle = percentage_off * 60
            odd_angle = 60 - even_angle
            
            # Calculate areas of even and odd-numbered zones
            even_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
            odd_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))

            # Sum of even_area and odd_area
            layer_4_area = even_area + odd_area
            layer_areas.append(layer_4_area)

            # Select color 2 from the palette that most closely matches the avg_lab
            color_2 = select_color_2(avg_lab, palette_lab, color_1, even_area, odd_area, bw_color, layer_areas[2] - layer_4_area)

            # Convert LAB colors to RGB for filling polygons
            color_1_rgb = (lab2rgb(clamp_lab(color_1).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]
            color_2_rgb = (lab2rgb(clamp_lab(color_2).reshape(1, 1, 3)) * 255).astype(np.uint8)[0][0]

            # Ensure RGB values are tuples of integers
            color_1_rgb = tuple(map(int, color_1_rgb))
            color_2_rgb = tuple(map(int, color_2_rgb))

            # Draw the 12 zones with alternating colors
            angle_offset = 30 - (even_angle / 2)  # Start from half the interior angle of the first zone
            for zone in range(12):
                if zone % 2 == 0:
                    angle = even_angle
                    angle_color = color_1_rgb
                else:
                    angle = odd_angle
                    angle_color = color_2_rgb

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

                # Clip the points to ensure they're not bigger than desired
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

        elif i == 3:  # Layer 3
            hex_radius = radius * (i / 7)
            hex_radius = int(hex_radius)
            layer_3_radius = hex_radius
            inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
            inner_coords = inner_hexagon.get_verts().astype(int)

            pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
            
            # Calculate and store the area of layer 5
            layer_3_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
            layer_areas.append(layer_3_area)

    return pattern

# Reshape the image into a 2D array of pixels
pixels = input_image.reshape(-1, 3)

# Apply K-means clustering to find the 16 best colors
num_palette_colors = 16
kmeans = KMeans(n_clusters=num_palette_colors, random_state=42).fit(pixels)
palette = kmeans.cluster_centers_

# Convert the palette to LAB color space
palette_lab = rgb2lab(palette.reshape(1, -1, 3) / 255.0).reshape(-1, 3)  # Normalize to [0, 1] range

# Sort the palette by their L values in the LAB color space
sorted_palette_indices = np.argsort(palette_lab[:, 0])
sorted_palette = palette[sorted_palette_indices]

# Create and save the palette image
palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)

for i, color in enumerate(sorted_palette):
    palette_image[:, i * 32:(i + 1) * 32] = color

plt.imsave('palette.png', palette_image)

# Create the hexagonal pattern grid
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
        hex_pattern = create_hex_pattern(center_x, center_y, hex_radius, avg_lab, palette_lab)

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
