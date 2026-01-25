#!/usr/bin/env python3
"""
Hexify CLI - Generate hexagonal pattern art from images.

Usage:
    hexify input.png -o output.png
    hexify input.png -c 16 --preset fast
    hexify input.png --config settings.yaml
    hexify batch inputs/ -o outputs/
    python -m v2.cli input.png -o output.png
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .settings import HexifySettings, ColorSpace, HexOrientation, QuantizationMethod
from .processor import HexagonProcessor
from .exceptions import HexifyError, InvalidImageError

# Version
__version__ = "2.1.0"

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Preset configurations
PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "num_palette_colors": 16,
        "num_layers": 7,
        "num_zones": 12,
        "chunk_size": 100,
    },
    "fast": {
        "num_palette_colors": 8,
        "num_layers": 5,
        "num_zones": 6,
        "chunk_size": 200,
        "quantization_method": "minibatch_kmeans",
    },
    "detailed": {
        "num_palette_colors": 32,
        "num_layers": 7,
        "num_zones": 18,
        "chunk_size": 50,
        "kmeans_n_init": 15,
    },
    "minimal": {
        "num_palette_colors": 6,
        "num_layers": 4,
        "num_zones": 6,
        "chunk_size": 200,
        "quantization_method": "minibatch_kmeans",
    },
}


class ColoredFormatter(logging.Formatter):
    """Logging formatter with color support."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Configure logging based on verbosity settings.

    Args:
        verbose: Enable debug-level logging
        quiet: Suppress all output except errors

    Returns:
        Configured logger
    """
    logger = logging.getLogger("hexify")

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler with colored output
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Use colors if terminal supports it
    if sys.stdout.isatty():
        formatter = ColoredFormatter("%(levelname)s: %(message)s")
    else:
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="hexify",
        description="Generate hexagonal pattern art from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hexify input.png                      Process with defaults
  hexify input.png -o output.png        Specify output path
  hexify input.png -c 16 --preset fast  Use fast preset with 16 colors
  hexify input.png --config my.yaml     Load settings from config file
  hexify inputs/ --batch -o outputs/    Batch process a directory
  hexify "*.jpg" --batch                Process all JPGs in current dir
        """,
    )

    # Input argument (required unless using info commands like --list-presets)
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input image path, directory (for batch mode), or glob pattern",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        help="Output path (default: input_hexified.png)",
    )
    output_group.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format (default: png)",
    )
    output_group.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="N",
        help="JPEG quality 1-100 (default: 95)",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "-c", "--colors",
        type=int,
        default=None,
        metavar="N",
        help="Number of palette colors (default: 16)",
    )
    proc_group.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: CPU count)",
    )
    proc_group.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Hexagons per processing chunk (default: 100)",
    )

    # Preset and config
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--preset",
        choices=["default", "fast", "detailed", "minimal"],
        default="default",
        help="Use a predefined preset (default: default)",
    )
    config_group.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="Load settings from YAML or JSON file",
    )
    config_group.add_argument(
        "--save-config",
        type=Path,
        metavar="FILE",
        help="Save current settings to YAML file and exit",
    )

    # Style options
    style_group = parser.add_argument_group("Style Options")
    style_group.add_argument(
        "--layers",
        type=int,
        default=None,
        metavar="N",
        help="Number of concentric layers (default: 7)",
    )
    style_group.add_argument(
        "--zones",
        type=int,
        default=None,
        metavar="N",
        help="Number of angular zones (default: 12)",
    )
    style_group.add_argument(
        "--border-width",
        type=int,
        default=None,
        metavar="N",
        help="Hexagon border width in pixels (default: 0)",
    )
    style_group.add_argument(
        "--border-color",
        type=str,
        default=None,
        metavar="R,G,B",
        help="Border color as R,G,B (e.g., 255,255,255)",
    )
    style_group.add_argument(
        "--orientation",
        choices=["flat_top", "pointy_top"],
        default=None,
        help="Hexagon orientation (default: flat_top)",
    )

    # Batch processing
    batch_group = parser.add_argument_group("Batch Processing")
    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in input directory",
    )
    batch_group.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Include subdirectories in batch mode",
    )
    batch_group.add_argument(
        "--pattern",
        type=str,
        default=None,
        metavar="GLOB",
        help="File pattern for batch mode (e.g., '*.png')",
    )

    # Output control
    output_ctrl_group = parser.add_argument_group("Output Control")
    output_ctrl_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug output",
    )
    output_ctrl_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )
    output_ctrl_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    output_ctrl_group.add_argument(
        "--save-hexagons",
        action="store_true",
        help="Save individual hexagon images",
    )

    # Info commands
    info_group = parser.add_argument_group("Information")
    info_group.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )
    info_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def parse_border_color(color_str: str) -> Tuple[int, int, int]:
    """
    Parse a border color string into RGB tuple.

    Args:
        color_str: Color as "R,G,B" string

    Returns:
        Tuple of (R, G, B) integers

    Raises:
        ValueError: If color string is invalid
    """
    try:
        parts = [int(x.strip()) for x in color_str.split(",")]
        if len(parts) != 3:
            raise ValueError("Must have exactly 3 components")
        for p in parts:
            if not 0 <= p <= 255:
                raise ValueError(f"Color value {p} out of range 0-255")
        return tuple(parts)
    except Exception as e:
        raise ValueError(f"Invalid border color '{color_str}': {e}")


def print_presets() -> None:
    """Print available presets and their configurations."""
    print("\nAvailable Presets:")
    print("-" * 60)
    for name, config in PRESETS.items():
        print(f"\n  {name}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    print()


def load_settings(args: argparse.Namespace, logger: logging.Logger) -> HexifySettings:
    """
    Load settings from config file and/or command line arguments.

    Priority (highest to lowest):
    1. Command line arguments
    2. Config file
    3. Preset
    4. Defaults

    Args:
        args: Parsed command line arguments
        logger: Logger instance

    Returns:
        Configured HexifySettings instance
    """
    settings_dict = {}

    # Start with preset settings
    if args.preset and args.preset in PRESETS:
        settings_dict.update(PRESETS[args.preset])
        logger.debug(f"Applied preset: {args.preset}")

    # Load config file if specified
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        config_path = str(args.config)
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            file_settings = HexifySettings.from_yaml(config_path)
        elif config_path.endswith(".json"):
            file_settings = HexifySettings.from_json(config_path)
        else:
            raise ValueError(f"Unknown config format: {args.config.suffix}")

        settings_dict.update(file_settings.to_dict())
        logger.debug(f"Loaded config from: {args.config}")

    # Override with command line arguments
    if args.colors is not None:
        settings_dict["num_palette_colors"] = args.colors
    if args.layers is not None:
        settings_dict["num_layers"] = args.layers
    if args.zones is not None:
        settings_dict["num_zones"] = args.zones
    if args.chunk_size is not None:
        settings_dict["chunk_size"] = args.chunk_size
    if args.border_width is not None:
        settings_dict["border_width"] = args.border_width
    if args.border_color is not None:
        settings_dict["border_color"] = parse_border_color(args.border_color)
    if args.orientation is not None:
        settings_dict["orientation"] = args.orientation

    # Convert string enum values
    if "quantization_method" in settings_dict and isinstance(settings_dict["quantization_method"], str):
        settings_dict["quantization_method"] = QuantizationMethod(settings_dict["quantization_method"])
    if "orientation" in settings_dict and isinstance(settings_dict["orientation"], str):
        settings_dict["orientation"] = HexOrientation(settings_dict["orientation"])
    if "color_space" in settings_dict and isinstance(settings_dict["color_space"], str):
        settings_dict["color_space"] = ColorSpace(settings_dict["color_space"])

    return HexifySettings.from_dict(settings_dict) if settings_dict else HexifySettings()


def get_default_output_path(input_path: Path, output_format: str) -> Path:
    """
    Generate default output path from input path.

    Args:
        input_path: Input file path
        output_format: Output format (png or jpg)

    Returns:
        Default output path (input_hexified.format)
    """
    stem = input_path.stem
    return input_path.parent / f"{stem}_hexified.{output_format}"


def discover_images(
    input_path: Path,
    recursive: bool = False,
    pattern: Optional[str] = None
) -> List[Path]:
    """
    Discover image files for batch processing.

    Args:
        input_path: Input path (file, directory, or glob pattern)
        recursive: Whether to search subdirectories
        pattern: Optional glob pattern to filter files

    Returns:
        List of image file paths
    """
    images = []

    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(input_path)
    elif input_path.is_dir():
        # Directory - scan for images
        if recursive:
            glob_pattern = "**/*" if pattern is None else f"**/{pattern}"
        else:
            glob_pattern = "*" if pattern is None else pattern

        for path in input_path.glob(glob_pattern):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    else:
        # Treat as glob pattern
        parent = input_path.parent if input_path.parent.exists() else Path(".")
        glob_str = input_path.name

        for path in parent.glob(glob_str):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

    return sorted(images)


def process_single(
    input_path: Path,
    output_path: Path,
    settings: HexifySettings,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[bool, float, Optional[float]]:
    """
    Process a single image file.

    Args:
        input_path: Input image path
        output_path: Output image path
        settings: Processing settings
        args: Command line arguments
        logger: Logger instance

    Returns:
        Tuple of (success, processing_time, cache_hit_rate)
    """
    start_time = time.time()

    try:
        # Load image
        logger.info(f"Loading: {input_path}")
        image = cv2.imread(str(input_path))
        if image is None:
            raise InvalidImageError(f"Failed to load image: {input_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.debug(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # Create processor
        processor = HexagonProcessor(
            num_palette_colors=settings.num_palette_colors,
            num_processes=args.jobs,
            chunk_size=settings.chunk_size,
            save_hexagons=args.save_hexagons,
        )

        # Setup progress bar if available and not disabled
        pbar = None
        if not args.no_progress and not args.quiet:
            try:
                from tqdm import tqdm
                # Estimate hexagon count based on grid setup
                processor.setup_hexagon_grid(image.shape)
                total_hexagons = len(processor.hex_centers)
                pbar = tqdm(total=total_hexagons, desc="Processing", unit="hex")
            except ImportError:
                logger.debug("tqdm not available, progress bar disabled")

        # Process image
        logger.info("Processing...")
        output = processor.process_image(image, pbar=pbar)

        if pbar:
            pbar.close()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save output
        logger.info(f"Saving: {output_path}")
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        if args.format == "jpg":
            cv2.imwrite(str(output_path), output_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            cv2.imwrite(str(output_path), output_bgr)

        elapsed = time.time() - start_time
        cache_hit_rate = processor.get_cache_hit_rate()

        logger.info(f"Complete in {elapsed:.2f}s (cache hit rate: {cache_hit_rate:.1%})")

        return True, elapsed, cache_hit_rate

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed: {e}")
        return False, elapsed, None


def process_batch(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Process multiple images in batch mode.

    Args:
        args: Command line arguments
        logger: Logger instance

    Returns:
        Exit code (0 for success)
    """
    input_path = Path(args.input)

    # Discover images
    images = discover_images(input_path, args.recursive, args.pattern)

    if not images:
        logger.error(f"No images found in: {input_path}")
        return 1

    logger.info(f"Found {len(images)} images to process")

    # Load settings
    settings = load_settings(args, logger)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        if input_path.is_dir():
            output_dir = input_path / "hexified"
        else:
            output_dir = input_path.parent / "hexified"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each image
    start_time = time.time()
    results = {
        "success": 0,
        "failed": 0,
        "total_time": 0.0,
        "cache_hits": 0.0,
        "cache_samples": 0,
    }

    # Overall progress bar
    pbar_overall = None
    if not args.no_progress and not args.quiet:
        try:
            from tqdm import tqdm
            pbar_overall = tqdm(total=len(images), desc="Batch", unit="img")
        except ImportError:
            pass

    for img_path in images:
        # Generate output path
        rel_path = img_path.relative_to(input_path.parent) if input_path.is_dir() else img_path.name
        output_path = output_dir / f"{rel_path.stem}_hexified.{args.format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Suppress individual progress bars in batch mode
        batch_args = argparse.Namespace(**vars(args))
        batch_args.no_progress = True

        success, elapsed, cache_rate = process_single(
            img_path, output_path, settings, batch_args, logger
        )

        if success:
            results["success"] += 1
            results["total_time"] += elapsed
            if cache_rate is not None:
                results["cache_hits"] += cache_rate
                results["cache_samples"] += 1
        else:
            results["failed"] += 1

        if pbar_overall:
            pbar_overall.update(1)

    if pbar_overall:
        pbar_overall.close()

    # Print summary
    total_time = time.time() - start_time
    avg_cache_rate = (
        results["cache_hits"] / results["cache_samples"]
        if results["cache_samples"] > 0
        else 0
    )

    print()
    print("=" * 50)
    print("Batch Processing Summary")
    print("=" * 50)
    print(f"  Images processed: {results['success']}")
    print(f"  Images failed:    {results['failed']}")
    print(f"  Total time:       {total_time:.2f}s")
    if results["success"] > 0:
        print(f"  Average time:     {results['total_time'] / results['success']:.2f}s per image")
        print(f"  Avg cache rate:   {avg_cache_rate:.1%}")
    print("=" * 50)

    return 0 if results["failed"] == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    logger = setup_logging(args.verbose, args.quiet)

    # Handle info commands first (these don't require input)
    if args.list_presets:
        print_presets()
        return 0

    # Handle save-config command (doesn't require input to exist)
    if args.save_config:
        try:
            settings = load_settings(args, logger)
            save_path = str(args.save_config)
            if save_path.endswith(".json"):
                settings.to_json(save_path)
            else:
                settings.to_yaml(save_path)
            print(f"Settings saved to: {args.save_config}")
            return 0
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return 1

    # Validate input is provided
    if args.input is None:
        logger.error("Input path is required")
        return 1

    input_path = Path(args.input)

    # Check if input exists (allow glob patterns)
    if not input_path.exists() and not any(input_path.parent.glob(input_path.name)):
        logger.error(f"Input not found: {args.input}")
        return 1

    # Determine if batch mode
    is_batch = args.batch or input_path.is_dir() or "*" in str(input_path)

    try:
        if is_batch:
            return process_batch(args, logger)
        else:
            # Single file mode
            settings = load_settings(args, logger)

            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = get_default_output_path(input_path, args.format)

            success, _, _ = process_single(
                input_path, output_path, settings, args, logger
            )
            return 0 if success else 1

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except HexifyError as e:
        logger.error(f"Processing error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
