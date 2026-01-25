"""Hexify - Transform images and videos into hexagonal pattern art.

This module provides command-line functionality for processing images
and videos through the hexagonal pattern filter, creating stylized
artistic outputs with customizable color palettes.

Example usage:
    python hexify.py image.png --colors 16 --processes 4
    python hexify.py video.mp4 --colors 32 --save-frames
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import traceback
from typing import TypeAlias, Any, TYPE_CHECKING

from hexagon_processor import HexagonProcessor
from tqdm import tqdm

# Type Aliases
ImageArray: TypeAlias = np.ndarray

# Video handlers imported only when needed (module may be missing)
VideoReader: type | None = None
VideoWriter: type | None = None
downscale_video: Any = None

# Supported file extensions
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


def create_output_directory(input_path: str) -> tuple[str, str, str]:
    """Create output directories for processed files.

    Creates a main output directory named after the input file,
    along with subdirectories for frames and hexagons.

    Args:
        input_path: Path to the input file.

    Returns:
        Tuple of (output_dir, frames_dir, hexagons_dir) paths.

    Raises:
        FileNotFoundError: If the input file does not exist.
        OSError: If directories cannot be created.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    base_name: str = os.path.splitext(os.path.basename(input_path))[0]
    parent_dir: str = os.path.dirname(input_path) or "."
    output_dir: str = os.path.join(parent_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    frames_dir: str = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    hexagons_dir: str = os.path.join(output_dir, "hexagons")
    os.makedirs(hexagons_dir, exist_ok=True)

    return output_dir, frames_dir, hexagons_dir


def validate_palette_colors(num_palette_colors: int) -> int:
    """Validate and adjust palette color count.

    Args:
        num_palette_colors: Requested number of palette colors.

    Returns:
        Validated palette color count (minimum 5, maximum 256).
    """
    if num_palette_colors < HexagonProcessor.MIN_PALETTE_COLORS:
        print(
            f"Warning: Palette size {num_palette_colors} is below minimum. "
            f"Setting to {HexagonProcessor.MIN_PALETTE_COLORS}."
        )
        return HexagonProcessor.MIN_PALETTE_COLORS
    if num_palette_colors > HexagonProcessor.MAX_PALETTE_COLORS:
        print(
            f"Warning: Palette size {num_palette_colors} exceeds maximum. "
            f"Setting to {HexagonProcessor.MAX_PALETTE_COLORS}."
        )
        return HexagonProcessor.MAX_PALETTE_COLORS
    return num_palette_colors


def process_image(
    input_image_path: str,
    num_palette_colors: int,
    num_processes: int | None,
    chunk_size: int = 32,
    save_hexagons: bool = True,
) -> None:
    """Process a single image through the hexagon filter.

    Loads an image, applies hexagonal pattern transformation with the
    specified color palette, and saves the output.

    Args:
        input_image_path: Path to the input image file.
        num_palette_colors: Number of colors for the palette (5-256).
        num_processes: Number of parallel processes. None for CPU count.
        chunk_size: Number of hexagons to process per chunk.
        save_hexagons: Whether to save individual hexagon images.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the image cannot be read or is invalid.
    """
    if not os.path.isfile(input_image_path):
        print(f"Error: The file '{input_image_path}' does not exist.")
        return

    output_dir, frames_dir, hexagons_dir = create_output_directory(input_image_path)

    input_image: ImageArray | None = cv2.imread(input_image_path)
    if input_image is None:
        print(
            f"Error: Unable to read the image file '{input_image_path}'. "
            "Please check if it's a valid image file."
        )
        return

    # Validate image dimensions
    height, width = input_image.shape[:2]
    if height < HexagonProcessor.MIN_IMAGE_SIZE:
        print(
            f"Error: Image height ({height}px) is below minimum "
            f"({HexagonProcessor.MIN_IMAGE_SIZE}px)."
        )
        return
    if width < HexagonProcessor.MIN_IMAGE_SIZE:
        print(
            f"Error: Image width ({width}px) is below minimum "
            f"({HexagonProcessor.MIN_IMAGE_SIZE}px)."
        )
        return

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    processor = HexagonProcessor(
        num_palette_colors, num_processes, hexagons_dir, chunk_size, save_hexagons
    )

    # Get the total number of hexagons
    processor.setup_hexagon_grid(input_image.shape)
    total_hexagons: int = len(processor.hex_centers)

    # Create a progress bar for image processing
    with tqdm(total=total_hexagons, desc="Processing image", unit="hexagon") as pbar:
        output_image: ImageArray = processor.process_image(input_image, pbar)

    # Save palette with new naming convention
    palette_image: ImageArray = np.zeros(
        (64, 32 * num_palette_colors, 3), dtype=np.uint8
    )
    for i, color in enumerate(processor.palette):
        palette_image[:, i * 32 : (i + 1) * 32] = color
    palette_filename: str = f"{processor.palette_hash[:6]}_palette.png"
    palette_path: str = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    # Save output image
    output_image_path: str = os.path.join(output_dir, "output.png")
    plt.imsave(output_image_path, output_image)
    print(f"Output image saved to: {output_image_path}")

    # Report cache hit rate and statistics
    cache_hit_rate: float = processor.get_cache_hit_rate()
    print(f"Cache hit rate: {cache_hit_rate:.2%}")
    print(f"Total cache hits: {processor.cache_hits}")
    print(f"Total cache misses: {processor.cache_misses}")
    print(f"Final cache size: {len(processor.hexagon_cache)}")


def process_video(
    input_video_path: str,
    num_palette_colors: int,
    num_processes: int | None,
    chunk_size: int = 32,
    save_hexagons: bool = True,
    save_frames: bool = True,
) -> None:
    """Process a video through the hexagon filter.

    Loads a video, applies hexagonal pattern transformation to each frame
    with a consistent color palette, and saves the output video.

    Args:
        input_video_path: Path to the input video file.
        num_palette_colors: Number of colors for the palette (5-256).
        num_processes: Number of parallel processes. None for CPU count.
        chunk_size: Number of hexagons to process per chunk.
        save_hexagons: Whether to save individual hexagon images.
        save_frames: Whether to save individual processed frames.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ImportError: If video_handlers module is not available.
    """
    global VideoReader, VideoWriter, downscale_video
    try:
        from video_handlers import (
            VideoReader,
            VideoWriter,
            downscale_video,
        )
    except ImportError:
        print("Error: video_handlers module not found. Video processing is not available.")
        print("Only image processing is supported with the current installation.")
        return

    if not os.path.isfile(input_video_path):
        print(f"Error: The file '{input_video_path}' does not exist.")
        return

    output_dir, frames_dir, hexagons_dir = create_output_directory(input_video_path)

    # Downscale video if necessary
    downscaled_path: str = os.path.join(output_dir, "downscaled.mp4")
    input_video_path = downscale_video(input_video_path, downscaled_path)

    reader = VideoReader(input_video_path)

    print(f"Total frames to process: {reader.frame_count}")

    # Generate palette from sample frames
    sample_frames: list[ImageArray] = reader.get_frames(num_frames=10)
    combined_image: ImageArray = np.concatenate(sample_frames, axis=1)

    processor = HexagonProcessor(
        num_palette_colors, num_processes, hexagons_dir, chunk_size, save_hexagons
    )
    processor.generate_palette(combined_image)

    # Save palette with new naming convention
    palette_image: ImageArray = np.zeros(
        (64, 32 * num_palette_colors, 3), dtype=np.uint8
    )
    for i, color in enumerate(processor.palette):
        palette_image[:, i * 32 : (i + 1) * 32] = color
    palette_filename: str = f"{processor.palette_hash[:6]}_palette.png"
    palette_path: str = os.path.join(output_dir, palette_filename)
    plt.imsave(palette_path, palette_image)
    print(f"Palette saved to: {palette_path}")

    # Process video
    output_video_path: str = os.path.join(output_dir, "output.mp4")
    writer = VideoWriter(
        output_video_path, reader.fps, reader.width * 4, reader.height * 4
    )

    failed_frames: list[int] = []

    # Setup hexagon grid to get total hexagons
    processor.setup_hexagon_grid((reader.height, reader.width))
    total_hexagons: int = len(processor.hex_centers)

    try:
        with tqdm(
            total=reader.frame_count, desc="Processing video frames", unit="frame"
        ) as frame_pbar:
            for frame_number in range(1, reader.frame_count + 1):
                try:
                    frame: ImageArray | None = reader.read_frame()
                    if frame is None:
                        print(f"Failed to read frame {frame_number}")
                        failed_frames.append(frame_number)
                        continue

                    with tqdm(
                        total=total_hexagons,
                        desc=f"Frame {frame_number}",
                        unit="hexagon",
                        leave=False,
                    ) as hexagon_pbar:
                        processed_frame: ImageArray = processor.process_image(
                            frame, hexagon_pbar
                        )

                    if save_frames:
                        # Save processed frame with palette hash in filename
                        frame_filename: str = (
                            f"{processor.palette_hash[:6]}_frame_{frame_number:06d}.png"
                        )
                        frame_path: str = os.path.join(frames_dir, frame_filename)
                        plt.imsave(frame_path, processed_frame)

                    downscaled_frame: ImageArray = cv2.resize(
                        processed_frame, (reader.width * 4, reader.height * 4)
                    )
                    writer.write_frame(downscaled_frame)

                    frame_pbar.update(1)

                    # Print cache hit rate every 10 frames
                    if frame_number % 10 == 0:
                        cache_hit_rate: float = processor.get_cache_hit_rate()
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
        final_cache_hit_rate: float = processor.get_cache_hit_rate()
        print(f"Final cache hit rate: {final_cache_hit_rate:.2%}")
        print(f"Total cache hits: {processor.cache_hits.value}")
        print(f"Total cache misses: {processor.cache_misses.value}")
        print(f"Final cache size: {len(processor.hexagon_cache)}")


def main(
    input_path: str,
    num_palette_colors: int = 16,
    num_processes: int | None = None,
    chunk_size: int = 32,
    save_hexagons: bool = True,
    save_frames: bool = True,
) -> None:
    """Main entry point for hexify processing.

    Determines the input file type and routes to the appropriate
    processing function.

    Args:
        input_path: Path to input image or video file.
        num_palette_colors: Number of colors for the palette (5-256).
        num_processes: Number of parallel processes. None for CPU count.
        chunk_size: Number of hexagons to process per chunk.
        save_hexagons: Whether to save individual hexagon images.
        save_frames: Whether to save individual processed frames (video only).

    Raises:
        ValueError: If the file format is not supported.
    """
    start_time: float = time.time()

    # Validate input path exists
    if not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    if input_path.lower().endswith(IMAGE_EXTENSIONS):
        process_image(
            input_path, num_palette_colors, num_processes, chunk_size, save_hexagons
        )
    elif input_path.lower().endswith(VIDEO_EXTENSIONS):
        process_video(
            input_path,
            num_palette_colors,
            num_processes,
            chunk_size,
            save_hexagons,
            save_frames,
        )
    else:
        print(f"Error: Unsupported file format for '{input_path}'")
        print(f"Supported image formats: {', '.join(IMAGE_EXTENSIONS)}")
        print(f"Supported video formats: {', '.join(VIDEO_EXTENSIONS)}")
        return

    end_time: float = time.time()
    total_time: float = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate hexagonal pattern from input image or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hexify.py image.png                    Process with default settings
  python hexify.py image.png -c 32              Use 32-color palette
  python hexify.py video.mp4 -p 8               Use 8 parallel processes
  python hexify.py image.png --save-hexagons    Save individual hexagon images
        """,
    )
    parser.add_argument(
        "input_path", help="Path to the input image or video file"
    )
    parser.add_argument(
        "-c",
        "--colors",
        type=int,
        default=16,
        help="Number of colors in the palette (default: 16, range: 5-256)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of hexagons to process in each chunk (default: 32)",
    )
    parser.add_argument(
        "--save-hexagons",
        action="store_true",
        default=False,
        help="Save individual hexagon images (default: False)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=False,
        help="Save individual processed frames from video (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Validate and adjust palette colors
    validated_colors: int = validate_palette_colors(args.colors)

    # Validate process count
    if args.processes is not None and args.processes < 1:
        print("Error: Number of processes must be at least 1.")
        exit(1)

    # Validate chunk size
    if args.chunk_size < 1:
        print("Error: Chunk size must be at least 1.")
        exit(1)

    main(
        args.input_path,
        validated_colors,
        args.processes,
        args.chunk_size,
        args.save_hexagons,
        args.save_frames,
    )
