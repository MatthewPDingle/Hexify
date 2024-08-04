import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from hexagon_processor import HexagonProcessor

def create_output_directory(input_image_path):
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_dir = os.path.join(os.path.dirname(input_image_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    hexagons_dir = os.path.join(output_dir, 'hexagons')
    os.makedirs(hexagons_dir, exist_ok=True)
    return output_dir, hexagons_dir

def main(input_image_path, num_palette_colors=16, num_processes=None):
    start_time = time.time() 
    
    # Check if the input file exists
    if not os.path.isfile(input_image_path):
        print(f"Error: The file '{input_image_path}' does not exist.")
        return

    output_dir, hexagons_dir = create_output_directory(input_image_path)
    
    # Read the image and handle potential errors
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Unable to read the image file '{input_image_path}'. Please check if it's a valid image file.")
        return

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    processor = HexagonProcessor(num_palette_colors, num_processes)
    output_image = processor.process_image(input_image)

    # Save palette
    palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)
    for i, color in enumerate(processor.palette):
        palette_image[:, i * 32:(i + 1) * 32] = color
    palette_filename = f"Palette - {processor.palette_hash[:6]}.png"
    palette_path = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    # Save output image
    output_image_path = os.path.join(output_dir, 'output.png')
    plt.imsave(output_image_path, output_image)
    print(f"Output image saved to: {output_image_path}")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
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