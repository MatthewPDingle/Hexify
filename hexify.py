import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import traceback
from hexagon_processor import HexagonProcessor
from video_handlers import VideoReader, VideoWriter, downscale_video
from tqdm import tqdm

def create_output_directory(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(os.path.dirname(input_path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    hexagons_dir = os.path.join(output_dir, 'hexagons')
    os.makedirs(hexagons_dir, exist_ok=True)
    return output_dir, frames_dir, hexagons_dir

def process_image(input_image_path, num_palette_colors, num_processes, chunk_size=32, save_hexagons=True):
    if not os.path.isfile(input_image_path):
        print(f"Error: The file '{input_image_path}' does not exist.")
        return

    output_dir, frames_dir, hexagons_dir = create_output_directory(input_image_path)
    
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Unable to read the image file '{input_image_path}'. Please check if it's a valid image file.")
        return

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    processor = HexagonProcessor(num_palette_colors, num_processes, hexagons_dir, chunk_size, save_hexagons)
    
    # Get the total number of hexagons
    processor.setup_hexagon_grid(input_image.shape)
    total_hexagons = len(processor.hex_centers)
    
    # Create a progress bar for image processing
    with tqdm(total=total_hexagons, desc="Processing image", unit="hexagon") as pbar:
        output_image = processor.process_image(input_image, pbar)

    # Save palette with new naming convention
    palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)
    for i, color in enumerate(processor.palette):
        palette_image[:, i * 32:(i + 1) * 32] = color
    palette_filename = f"{processor.palette_hash[:6]}_palette.png"
    palette_path = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    # Save output image
    output_image_path = os.path.join(output_dir, 'output.png')
    plt.imsave(output_image_path, output_image)
    print(f"Output image saved to: {output_image_path}")

    # Report cache hit rate and statistics
    cache_hit_rate = processor.get_cache_hit_rate()
    print(f"Cache hit rate: {cache_hit_rate:.2%}")
    print(f"Total cache hits: {processor.cache_hits}")
    print(f"Total cache misses: {processor.cache_misses}")
    print(f"Final cache size: {len(processor.hexagon_cache)}")

def process_video(input_video_path, num_palette_colors, num_processes, chunk_size=32, save_hexagons=True, save_frames=True):
    if not os.path.isfile(input_video_path):
        print(f"Error: The file '{input_video_path}' does not exist.")
        return

    output_dir, frames_dir, hexagons_dir = create_output_directory(input_video_path)

    # Downscale video if necessary
    downscaled_path = os.path.join(output_dir, 'downscaled.mp4')
    input_video_path = downscale_video(input_video_path, downscaled_path)

    reader = VideoReader(input_video_path)
    
    print(f"Total frames to process: {reader.frame_count}")
    
    # Generate palette from sample frames
    sample_frames = reader.get_frames(num_frames=10)
    combined_image = np.concatenate(sample_frames, axis=1)
    
    processor = HexagonProcessor(num_palette_colors, num_processes, hexagons_dir, chunk_size, save_hexagons)
    processor.generate_palette(combined_image)

    # Save palette with new naming convention
    palette_image = np.zeros((64, 32 * num_palette_colors, 3), dtype=np.uint8)
    for i, color in enumerate(processor.palette):
        palette_image[:, i * 32:(i + 1) * 32] = color
    palette_filename = f"{processor.palette_hash[:6]}_palette.png"
    palette_path = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    # Process video
    output_video_path = os.path.join(output_dir, 'output.mp4')
    writer = VideoWriter(output_video_path, reader.fps, reader.width * 4, reader.height * 4)

    failed_frames = []

    # Setup hexagon grid to get total hexagons
    processor.setup_hexagon_grid((reader.height, reader.width))
    total_hexagons = len(processor.hex_centers)

    try:
        with tqdm(total=reader.frame_count, desc="Processing video frames", unit="frame") as frame_pbar:
            for frame_number in range(1, reader.frame_count + 1):
                try:
                    frame = reader.read_frame()
                    if frame is None:
                        print(f"Failed to read frame {frame_number}")
                        failed_frames.append(frame_number)
                        continue
                    
                    with tqdm(total=total_hexagons, desc=f"Frame {frame_number}", unit="hexagon", leave=False) as hexagon_pbar:
                        processed_frame = processor.process_image(frame, hexagon_pbar)
                    
                    if save_frames:
                        # Save processed frame with palette hash in filename
                        frame_filename = f"{processor.palette_hash[:6]}_frame_{frame_number:06d}.png"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        plt.imsave(frame_path, processed_frame)
                    
                    downscaled_frame = cv2.resize(processed_frame, (reader.width * 4, reader.height * 4))
                    writer.write_frame(downscaled_frame)
                    
                    frame_pbar.update(1)
                    
                    # Print cache hit rate every 10 frames
                    if frame_number % 10 == 0:
                        cache_hit_rate = processor.get_cache_hit_rate()
                        print(f"Current cache hit rate: {cache_hit_rate:.2%}")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_number}: {str(e)}")
                    print(traceback.format_exc())
                    failed_frames.append(frame_number)
    except Exception as e:
        print(f"An error occurred during video processing: {str(e)}")
        print(traceback.format_exc())
    finally:
        reader.close()
        writer.close()
        print(f"Processed video saved to: {output_video_path}")
        print(f"Final frame count: {reader.frame_count}")
        if failed_frames:
            print(f"Failed frames: {failed_frames}")
        
        # Report final cache hit rate
        final_cache_hit_rate = processor.get_cache_hit_rate()
        print(f"Final cache hit rate: {final_cache_hit_rate:.2%}")
        print(f"Total cache hits: {processor.cache_hits.value}")
        print(f"Total cache misses: {processor.cache_misses.value}")
        print(f"Final cache size: {len(processor.hexagon_cache)}")

def main(input_path, num_palette_colors=16, num_processes=None, chunk_size=32, save_hexagons=True, save_frames=True):
    start_time = time.time()

    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(input_path, num_palette_colors, num_processes, chunk_size, save_hexagons)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_path, num_palette_colors, num_processes, chunk_size, save_hexagons, save_frames)
    else:
        print(f"Error: Unsupported file format for '{input_path}'")
        return

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hexagonal pattern from input image or video.")
    parser.add_argument("input_path", help="Path to the input image or video file")
    parser.add_argument("-c", "--colors", type=int, default=16, help="Number of colors in the palette (default: 16)")
    parser.add_argument("-p", "--processes", type=int, default=None, 
                        help="Number of processes to use (default: number of CPU cores)")
    parser.add_argument("--chunk-size", type=int, default=32, 
                        help="Number of hexagons to process in each chunk (default: 32)")
    parser.add_argument("--save-hexagons", action="store_true", default=False,
                        help="Save individual hexagon images (default: False)")
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save individual processed frames from video (default: False)")
    
    args = parser.parse_args()
    
    if args.colors < 5:
        print("Minimum color palette size must be at least 5. Setting it to 5.")
        args.colors = 5
    
    main(args.input_path, args.colors, args.processes, args.chunk_size, args.save_hexagons, args.save_frames)