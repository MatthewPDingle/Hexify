import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from sklearn.cluster import KMeans
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

# Utility functions
def average_color(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    avg_color = masked[np.where(mask != 0)].mean(axis=0)
    return np.round(avg_color).astype(int)

def create_hex_mask(center_x, center_y, radius, shape, bw_rgb=0):
    if bw_rgb == 0:
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        mask = np.full(shape, bw_rgb, dtype=np.uint8)

    hexagon = RegularPolygon((center_x, center_y), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts()
    coords = np.clip(coords, [0, 0], [shape[1] - 1, shape[0] - 1]).astype(int)
    mask = cv2.fillPoly(mask, [coords], 255)
    return mask

def clip_point_to_hexagon(point, hex_coords):
    path = Path(hex_coords)
    if path.contains_point(point):
        return point

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

# Color selection functions
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

def select_color_2(avg_rgb, palette_rgb, color_1, color_1_area, color_2_area, bw_color, bw_color_area):
    min_distance = float('inf')
    best_color_2 = None
    
    color_1 = np.array(color_1)
    bw_color = np.array(bw_color)
    
    for color in palette_rgb:
        if np.array_equal(color, color_1):
            continue
        
        color = np.array(color)
        
        mixed_rgb = (color_1_area * color_1 + color_2_area * color + bw_color_area * bw_color) / (color_1_area + color_2_area + bw_color_area)
        
        distance = np.linalg.norm(avg_rgb - mixed_rgb)
        
        if distance < min_distance:
            min_distance = distance
            best_color_2 = color
    
    return best_color_2

# Hexagon pattern functions
def fill_odd_layer(pattern, radius, hex_radius, bw_color):
    inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
    inner_coords = inner_hexagon.get_verts().astype(int)
    pattern = cv2.fillPoly(pattern, [inner_coords], tuple(map(int, bw_color)))
    hex_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
    return pattern, hex_area

def fill_even_layer(i, avg_rgb, palette_rgb, radius, layer_radii, avoid_rgb, layer_areas, pattern, input_image, center_x, center_y, bw_color):
    brightness = np.mean(avg_rgb)
    if i == 6:
        diameter = 256 - abs(brightness - 128) * (256 - 192) / 128
    else:
        diameter = (layer_radii[i + 1] * 2) - abs(brightness - 128) * (64) / 128
    hex_radius = int(diameter // 2)
    inner_hexagon = RegularPolygon((radius, radius), numVertices=6, radius=hex_radius, orientation=np.pi / 2)
    inner_coords = inner_hexagon.get_verts().astype(int)

    mask = create_hex_mask(center_x, center_y, hex_radius, pattern.shape[:2])
    scaled_center_x = center_x // 16
    scaled_center_y = center_y // 16
    scaled_radius = hex_radius // 16

    input_mask = create_hex_mask(scaled_center_x, scaled_center_y, scaled_radius, input_image.shape[:2])
    avg_rgb = average_color(input_image, input_mask)

    color_1 = closest_palette_color(avg_rgb, palette_rgb, avoid_rgb)
    avoid_rgb.append(color_1)
    dist_1 = np.linalg.norm(color_1 - avg_rgb)

    color_adj = min(palette_rgb, key=lambda color: np.linalg.norm(color - color_1) if not np.array_equal(color, color_1) else float('inf'))
    dist_adj = np.linalg.norm(color_1 - color_adj)

    percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj) if dist_adj != 0 else 0
    even_angle = percentage_off * 60
    odd_angle = 60 - even_angle

    even_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(even_angle))
    odd_area = 6 * (0.5 * hex_radius * hex_radius * math.radians(odd_angle))

    layer_area = even_area + odd_area

    color_2 = select_color_2(avg_rgb, palette_rgb, color_1, even_area, odd_area, bw_color, layer_areas[i + 1] - layer_area)
    avoid_rgb.append(color_2)

    color_1_rgb = tuple(map(int, color_1))
    color_2_rgb = tuple(map(int, color_2))

    angle_offset = 30 - (even_angle / 2)
    for zone in range(12):
        angle = even_angle if zone % 2 == 0 else odd_angle
        angle_color = color_1_rgb if zone % 2 == 0 else color_2_rgb

        start_angle = angle_offset
        end_angle = start_angle + angle
        angle_offset += angle

        p1 = (radius + hex_radius * math.cos(math.radians(start_angle)), radius + hex_radius * math.sin(math.radians(start_angle)))
        p2 = (radius + hex_radius * math.cos(math.radians(end_angle)), radius + hex_radius * math.sin(math.radians(end_angle)))

        p12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5 / 2

        length_of_shorter_leg = d / math.sqrt(3)

        dx = p12[0] - p1[0]
        dy = p12[1] - p1[1]

        length = (dx ** 2 + dy ** 2) ** 0.5
        dx /= length
        dy /= length

        p3 = (p12[0] + length_of_shorter_leg * dy, p12[1] - length_of_shorter_leg * dx)

        p1 = clip_point_to_hexagon(p1, inner_coords)
        p2 = clip_point_to_hexagon(p2, inner_coords)
        p3 = clip_point_to_hexagon(p3, inner_coords)

        vertices1 = [(radius, radius), p1, p2]
        vertices1 = np.array(vertices1, dtype=np.int32)
        cv2.fillPoly(pattern, [vertices1], angle_color)

        if zone % 2 == 1:
            vertices2 = [p1, p2, p3]
            vertices2 = np.array(vertices2, dtype=np.int32)
            cv2.fillPoly(pattern, [vertices2], angle_color)

    return pattern, layer_area

def create_hex_pattern(center_x, center_y, radius, avg_rgb, palette_rgb, input_image):
    bw_rgb = 0 if np.mean(avg_rgb) < 128 else 255
    pattern = np.zeros((2 * radius, 2 * radius, 3), dtype=np.uint8)
    if bw_rgb != 0:
        pattern = np.full((2 * radius, 2 * radius, 3), bw_rgb, dtype=np.uint8)
    hexagon = RegularPolygon((radius, radius), numVertices=6, radius=radius, orientation=np.pi / 2)
    coords = hexagon.get_verts().astype(int)

    layer_areas = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}
    layer_radii = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0, 1: 0}
    
    bw_color = (0, 0, 0) if np.mean(avg_rgb) < 128 else (255, 255, 255)
    avoid_rgb = []

    for i in range(7, 0, -1):
        if i % 2 == 1:
            hex_radius = int(radius * (i / 7))
            layer_radii[i] = hex_radius
            pattern, layer_area = fill_odd_layer(pattern, radius, hex_radius, bw_color)
            layer_areas[i] = layer_area
        elif i % 2 == 0:
            pattern, layer_area = fill_even_layer(i, avg_rgb, palette_rgb, radius, layer_radii, avoid_rgb, layer_areas, pattern, input_image, center_x, center_y, bw_color)
            layer_areas[i] = layer_area

    return pattern

def process_hexagon(args):
    center_x, center_y, hex_radius, output_shape, input_image, sorted_palette, palette_hash, hexagon_cache, hexagons_dir = args
    if 0 <= center_x - hex_radius < output_shape[1] and 0 <= center_y - hex_radius < output_shape[0]:      
        input_center_x = int(center_x / 16)
        input_center_y = int(center_y / 16)
        input_hex_radius = hex_radius // 16

        input_mask = create_hex_mask(input_center_x, input_center_y, input_hex_radius, input_image.shape[:2])
        avg_rgb = average_color(input_image, input_mask)
        avg_rgb_key = tuple(avg_rgb)

        bw_rgb = 0 if np.mean(avg_rgb) < 128 else 255
        mask = create_hex_mask(center_x, center_y, hex_radius, output_shape[:2])
        #print(center_x, center_y)

        if avg_rgb_key in hexagon_cache:
            hex_pattern = hexagon_cache[avg_rgb_key]
            cache_hit = True
        else:
            hex_pattern = create_hex_pattern(center_x, center_y, hex_radius, avg_rgb, sorted_palette, input_image)
            hexagon_cache[avg_rgb_key] = hex_pattern
            cache_hit = False

            # Save unique hexagon
            hexagon_filename = f"hexagon_{palette_hash[:6]}_{avg_rgb[0]:03d}_{avg_rgb[1]:03d}_{avg_rgb[2]:03d}.png"
            hexagon_path = os.path.join(hexagons_dir, hexagon_filename)
            if not os.path.exists(hexagon_path):
                plt.imsave(hexagon_path, hex_pattern)

        # Passing the mask the black or white color of the hexagon fixes the aliasing bug
        mask_pattern = create_hex_mask(hex_radius, hex_radius, hex_radius, hex_pattern.shape[:2], bw_rgb)
        hex_pattern_masked = cv2.bitwise_and(hex_pattern, hex_pattern, mask=mask_pattern)

        x_start = max(center_x - hex_radius, 0)
        y_start = max(center_y - hex_radius, 0)
        x_end = min(center_x + hex_radius, output_shape[1])
        y_end = min(center_y + hex_radius, output_shape[0])

        return (x_start, y_start, x_end, y_end, hex_pattern_masked, mask[y_start:y_end, x_start:x_end], cache_hit)
    return None

def create_output_directory(input_image_path):
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_dir = os.path.join(os.path.dirname(input_image_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    hexagons_dir = os.path.join(output_dir, 'hexagons')
    os.makedirs(hexagons_dir, exist_ok=True)
    return output_dir, hexagons_dir

def main(input_image_path, num_palette_colors=16, num_processes=None):
    start_time = time.time() 
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    output_dir, hexagons_dir = create_output_directory(input_image_path)
    
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    output_image_shape = (input_image.shape[0] * 16, input_image.shape[1] * 16, 3)
    output_image = np.zeros(output_image_shape, dtype=np.uint8)

    hex_diameter = 256
    hex_radius = hex_diameter // 2
    hex_width = 2 * hex_radius
    #hex_height = math.sqrt((hex_radius * hex_radius) - (hex_radius / 2 * hex_radius / 2)) * 2
    hex_height = hex_width * (math.sqrt(3)/2)
    cols = int(output_image.shape[1] // (hex_width - (hex_width / 4))) + 1
    rows = int(output_image.shape[0] // (hex_height * 0.75)) + 1
    rows = int(rows * .75)

    hex_centers = [
        (
            int((hex_width - 64) * col) - 128,
            int(hex_height * row + (0.5 * hex_height if col % 2 else 0)) - 128
        )
        for row in range(rows)
        for col in range(cols)
    ]

    pixels = input_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_palette_colors, random_state=42).fit(pixels)
    palette = kmeans.cluster_centers_

    sorted_palette_indices = np.argsort(np.mean(palette, axis=1))
    sorted_palette = palette[sorted_palette_indices]

    # Generate palette hash
    palette_hash = hashlib.sha256(sorted_palette.tobytes()).hexdigest()

    palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)
    for i, color in enumerate(sorted_palette):
        palette_image[:, i * 32:(i + 1) * 32] = color
    palette_filename = f"Palette - {palette_hash}.png"
    palette_path = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    total_hexes = len(hex_centers)
    
    # Use a Manager to create a shared cache
    with multiprocessing.Manager() as manager:
        hexagon_cache = manager.dict()
        cache_hits = manager.Value('i', 0)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_hexagon, (center_x, center_y, hex_radius, output_image_shape, input_image, sorted_palette, palette_hash, hexagon_cache, hexagons_dir)) 
                       for center_x, center_y in hex_centers]
            
            for future in tqdm(as_completed(futures), total=total_hexes, desc="Processing hexagons"):
                result = future.result()
                if result:
                    x_start, y_start, x_end, y_end, hex_pattern_masked, mask_slice, cache_hit = result
                    hex_slice = output_image[y_start:y_end, x_start:x_end]
                    hex_slice[mask_slice != 0] = hex_pattern_masked[mask_slice != 0]
                    if cache_hit:
                        cache_hits.value += 1

        print("\nProcessing complete.")
        
        # Performance metrics
        unique_hexagons = len(hexagon_cache)
        total_hexagons = total_hexes
        cache_hit_rate = (cache_hits.value / total_hexagons) * 100
        estimated_time_saved = (cache_hits.value * 0.1)  # Assuming each hexagon takes about 0.1 seconds to process

        print(f"Number of unique hexagons: {unique_hexagons}")
        print(f"Total number of hexagons: {total_hexagons}")
        print(f"Cache hit rate: {cache_hit_rate:.2f}%")
        print(f"Estimated time saved by caching: {estimated_time_saved:.2f} seconds")

    output_image_path = os.path.join(output_dir, 'output.png')
    plt.imsave(output_image_path, output_image)
    print(f"Output image saved to: {output_image_path}")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate hexagonal pattern from input image.")
    parser.add_argument("input_image_path", help="Path to the input image file")
    parser.add_argument("-c", "--colors", type=int, default=16, help="Number of colors in the palette (default: 16)")
    parser.add_argument("-p", "--processes", type=int, default=None, 
                        help="Number of processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    if args.colors < 5:
        print("Minimum color palette size must be at least 5. Setting it to 5.")
        args.colors = 5
    
    main(args.input_image_path, args.colors, args.processes)