import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from sklearn.cluster import KMeans
import os
import time

def main(input_image_path, num_palette_colors=16):
    start_time = time.time() 
    
    # Load input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # 16x the resolution of the output image
    output_image_shape = (input_image.shape[0] * 16, input_image.shape[1] * 16, 3)
    output_image = np.zeros(output_image_shape, dtype=np.uint8)

    # Hexagon parameters with a diameter of 256 pixels
    hex_diameter = 256
    hex_radius = hex_diameter // 2
    hex_width = 2 * hex_radius
    hex_height = math.sqrt((hex_radius * hex_radius) - (hex_radius / 2 * hex_radius / 2)) * 2
    cols = int(output_image.shape[1] // (hex_width - (hex_width / 4))) + 1
    rows = int(output_image.shape[0] // (hex_height * 0.75)) + 1
    rows = int(rows * .75)

    # Create a list of hexagon centers in a staggered grid
    hex_centers = [
        (
            int((hex_width - 64) * col) - 128,
            int(hex_height * 1 * row + (0.5 * hex_height if col % 2 else 0)) - 128
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

    # Function to find the closest color in the palette to a given RGB color
    def closest_palette_color(avg_rgb, palette_rgb, avoid_rgb=None):
        min_distance = float('inf')
        closest_color = None

        for color in palette_rgb:
            if avoid_rgb is not None and any(np.array_equal(color, avoid_color) for avoid_color in avoid_rgb):
                continue
            
            distance = np.linalg.norm(avg_rgb - color)
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        
        return closest_color

    # Function to select a second color for the pattern
    def select_color_2(avg_rgb, palette_rgb, color_1, color_1_area, color_2_area, bw_color, bw_color_area):
        min_distance = float('inf')
        best_color_2 = None
        
        color_1 = np.array(color_1)
        bw_color = np.array(bw_color)
        
        for color in palette_rgb:
            if np.array_equal(color, color_1):
                continue
            
            color = np.array(color)
            
            # Calculate the mixed RGB color
            mixed_rgb = (color_1_area * color_1 + color_2_area * color + bw_color_area * bw_color) / (color_1_area + color_2_area + bw_color_area)
            
            # Calculate the distance to the average RGB color
            distance = np.linalg.norm(avg_rgb - mixed_rgb)
            
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

    # Function to find the nearest point on a segment
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

    # Helper function to fill in odd layers
    def fill_odd_layer(pattern, radius, hex_radius, bw_color):
        inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
        inner_coords = inner_hexagon.get_verts().astype(int)
        pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
        hex_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
        return pattern, hex_area

    # Helper function to fill in even layers
    def fill_even_layer(i, avg_rgb, palette_rgb, radius, layer_radii, avoid_rgb, layer_areas, pattern, input_image, center_x, center_y, bw_color):
        brightness = np.mean(avg_rgb)
        if i == 6:
            diameter = 256 - abs(brightness - 128) * (256 - 192) / 128
        else:
            diameter = (layer_radii[i + 1] * 2) - abs(brightness - 128) * (64) / 128
        hex_radius = int(diameter // 2)
        inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
        inner_coords = inner_hexagon.get_verts().astype(int)

        # Scale the input image coordinates to the output image coordinates
        mask = create_hex_mask(center_x, center_y, hex_radius, pattern.shape[:2])
        scaled_center_x = center_x // 16
        scaled_center_y = center_y // 16
        scaled_radius = hex_radius // 16

        input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
        avg_rgb = average_color(input_image, input_mask)

        # Select the color from the palette that most closely matches the avg_rgb
        color_1 = closest_palette_color(avg_rgb, palette_rgb, avoid_rgb)
        avoid_rgb.append(color_1)
        dist_1 = np.linalg.norm(color_1 - avg_rgb)

        # Find the next color from the palette that is most similar to color_1
        color_adj = min(palette_rgb, key=lambda color: np.linalg.norm(color - color_1) if not np.array_equal(color, color_1) else float('inf'))
        dist_adj = np.linalg.norm(color_1 - color_adj)

        # Determine the percentage dist_1 is of dist_adj
        percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
        even_angle = percentage_off * 60
        odd_angle = 60 - even_angle

        # Calculate areas of even and odd-numbered zones
        even_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
        odd_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))

        # Sum of even_area and odd_area
        layer_area = even_area + odd_area

        # Select color 2 from the palette that most closely matches the avg_rgb
        color_2 = select_color_2(avg_rgb, palette_rgb, color_1, even_area, odd_area, bw_color, layer_areas[i + 1] - layer_area)
        avoid_rgb.append(color_2)

        # Convert RGB colors to tuples for filling polygons
        color_1_rgb = tuple(map(int, color_1))
        color_2_rgb = tuple(map(int, color_2))

        # Draw the 12 zones with alternating colors
        angle_offset = 30 - (even_angle / 2)  # Start from half the interior angle of the first zone
        for zone in range(12):
            angle = even_angle if zone % 2 == 0 else odd_angle
            angle_color = color_1_rgb if zone % 2 == 0 else color_2_rgb

            start_angle = angle_offset
            end_angle = start_angle + angle
            angle_offset += angle

            # Calculate the vertices of p1, p2, and p3
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

        return pattern, layer_area

    # Function to create concentric hexagons with patterns
    def create_hex_pattern(center_x, center_y, radius, avg_rgb, palette_rgb):
        pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
        hexagon = RegularPolygon((radius, radius), numVertices=6, radius=radius, orientation=np.pi / 2)
        coords = hexagon.get_verts().astype(int)

        layer_areas = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}
        layer_radii = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}
        
        bw_color = (0, 0, 0) if np.mean(avg_rgb) < 128 else (255, 255, 255)
        avoid_rgb = []

        for i in range(7, 0, -1):  # Iterate from outer to inner layer
            if i % 2 == 1: # Odd layers
                hex_radius = int(radius * (i / 7))
                layer_radii[i] = hex_radius
                pattern, layer_area = fill_odd_layer(pattern, radius, hex_radius, bw_color)
                layer_areas[i] = layer_area
            elif i % 2 == 0: # Even layers
                pattern, layer_area = fill_even_layer(i, avg_rgb, palette_rgb, radius, layer_radii, avoid_rgb, layer_areas, pattern, input_image, center_x, center_y, bw_color)
                layer_areas[i] = layer_area

        return pattern

    # Reshape the image into a 2D array of pixels
    pixels = input_image.reshape(-1, 3)

    # Apply K-means clustering to find the best colors
    kmeans = KMeans(n_clusters=num_palette_colors, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_

    # Sort the palette by their brightness values
    sorted_palette_indices = np.argsort(np.mean(palette, axis=1))
    sorted_palette = palette[sorted_palette_indices]

    # Create and save the palette image
    palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)

    for i, color in enumerate(sorted_palette):
        palette_image[:, i * 32:(i + 1) * 32] = color

    plt.imsave('palette.png', palette_image)

    # Create the hexagonal pattern grid
    total_hexes = len(hex_centers)
    for i, (center_x, center_y) in enumerate(hex_centers):
        if 0 <= center_x - hex_radius < output_image.shape[1] and 0 <= center_y - hex_radius < output_image.shape[0]:
            # Scale the input image coordinates to the output image coordinates
            mask = create_hex_mask(center_x, center_y, hex_radius, output_image.shape[:2])
            scaled_center_x = int(center_x / 16)
            scaled_center_y = int(center_y / 16)
            scaled_radius = hex_radius // 16

            input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
            avg_rgb = average_color(input_image, input_mask)

            # Create a pattern using the average RGB color
            hex_pattern = create_hex_pattern(center_x, center_y, hex_radius, avg_rgb, sorted_palette)

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

        # Print progress
        progress = (i + 1) / total_hexes * 100
        sys.stdout.write(f"\rProgress: {progress:.2f}%")
        sys.stdout.flush()

    print("\nProcessing complete.")

    # Save and display the output image
    output_image_path = os.path.splitext(input_image_path)[0] + '_out.png'
    plt.imsave(output_image_path, output_image)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_image_path> [num_palette_colors]")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    num_palette_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if num_palette_colors < 5:
        print("Minimum color palette size must be at least 5. Setting it to 5.")
        num_palette_colors = 5
    main(input_image_path, num_palette_colors)