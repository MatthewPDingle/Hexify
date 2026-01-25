"""Tests for the Hexify CLI."""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if PyYAML is available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from v2.cli import (
    PRESETS,
    IMAGE_EXTENSIONS,
    create_parser,
    parse_border_color,
    get_default_output_path,
    discover_images,
    load_settings,
    setup_logging,
    print_presets,
    main,
)
from v2.settings import HexifySettings, QuantizationMethod, HexOrientation


class TestArgumentParser:
    """Test argument parsing."""

    def test_create_parser_returns_parser(self):
        """create_parser should return an ArgumentParser instance."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_input_optional_for_info_commands(self):
        """Parser should allow no input for info commands."""
        parser = create_parser()
        # This should not raise an error
        args = parser.parse_args(["--list-presets"])
        assert args.list_presets is True
        assert args.input is None

    def test_parser_accepts_input(self):
        """Parser should accept an input path."""
        parser = create_parser()
        args = parser.parse_args(["input.png"])
        assert args.input == "input.png"

    def test_parser_output_option(self):
        """Parser should accept -o/--output option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "-o", "output.png"])
        assert args.output == "output.png"

        args = parser.parse_args(["input.png", "--output", "out.png"])
        assert args.output == "out.png"

    def test_parser_colors_option(self):
        """Parser should accept -c/--colors option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "-c", "24"])
        assert args.colors == 24

        args = parser.parse_args(["input.png", "--colors", "32"])
        assert args.colors == 32

    def test_parser_jobs_option(self):
        """Parser should accept -j/--jobs option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "-j", "4"])
        assert args.jobs == 4

        args = parser.parse_args(["input.png", "--jobs", "8"])
        assert args.jobs == 8

    def test_parser_preset_option(self):
        """Parser should accept --preset option."""
        parser = create_parser()

        for preset in ["default", "fast", "detailed", "minimal"]:
            args = parser.parse_args(["input.png", "--preset", preset])
            assert args.preset == preset

    def test_parser_preset_invalid(self):
        """Parser should reject invalid preset names."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["input.png", "--preset", "invalid"])

    def test_parser_format_option(self):
        """Parser should accept --format option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "--format", "png"])
        assert args.format == "png"

        args = parser.parse_args(["input.png", "--format", "jpg"])
        assert args.format == "jpg"

    def test_parser_format_invalid(self):
        """Parser should reject invalid format."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["input.png", "--format", "gif"])

    def test_parser_batch_options(self):
        """Parser should accept batch processing options."""
        parser = create_parser()

        args = parser.parse_args(["input/", "--batch"])
        assert args.batch is True

        args = parser.parse_args(["input/", "--recursive"])
        assert args.recursive is True

        args = parser.parse_args(["input/", "-r"])
        assert args.recursive is True

        args = parser.parse_args(["input/", "--pattern", "*.png"])
        assert args.pattern == "*.png"

    def test_parser_style_options(self):
        """Parser should accept style options."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "--layers", "5"])
        assert args.layers == 5

        args = parser.parse_args(["input.png", "--zones", "18"])
        assert args.zones == 18

        args = parser.parse_args(["input.png", "--border-width", "2"])
        assert args.border_width == 2

        args = parser.parse_args(["input.png", "--border-color", "255,255,255"])
        assert args.border_color == "255,255,255"

        args = parser.parse_args(["input.png", "--orientation", "pointy_top"])
        assert args.orientation == "pointy_top"

    def test_parser_output_control_options(self):
        """Parser should accept output control options."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "-v"])
        assert args.verbose is True

        args = parser.parse_args(["input.png", "--verbose"])
        assert args.verbose is True

        args = parser.parse_args(["input.png", "-q"])
        assert args.quiet is True

        args = parser.parse_args(["input.png", "--quiet"])
        assert args.quiet is True

        args = parser.parse_args(["input.png", "--no-progress"])
        assert args.no_progress is True

    def test_parser_config_option(self):
        """Parser should accept --config option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "--config", "settings.yaml"])
        assert args.config == Path("settings.yaml")

    def test_parser_save_config_option(self):
        """Parser should accept --save-config option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "--save-config", "output.yaml"])
        assert args.save_config == Path("output.yaml")

    def test_parser_list_presets_option(self):
        """Parser should accept --list-presets option."""
        parser = create_parser()

        args = parser.parse_args(["input.png", "--list-presets"])
        assert args.list_presets is True

    def test_parser_defaults(self):
        """Parser should have correct defaults."""
        parser = create_parser()
        args = parser.parse_args(["input.png"])

        assert args.output is None
        assert args.format == "png"
        assert args.quality == 95
        assert args.colors is None
        assert args.jobs is None
        assert args.chunk_size is None
        assert args.preset == "default"
        assert args.config is None
        assert args.batch is False
        assert args.recursive is False
        assert args.verbose is False
        assert args.quiet is False
        assert args.no_progress is False


class TestBorderColorParsing:
    """Test border color parsing."""

    def test_parse_valid_color(self):
        """parse_border_color should parse valid color strings."""
        assert parse_border_color("0,0,0") == (0, 0, 0)
        assert parse_border_color("255,255,255") == (255, 255, 255)
        assert parse_border_color("128,64,32") == (128, 64, 32)

    def test_parse_color_with_spaces(self):
        """parse_border_color should handle spaces."""
        assert parse_border_color("0, 0, 0") == (0, 0, 0)
        assert parse_border_color(" 255 , 128 , 64 ") == (255, 128, 64)

    def test_parse_invalid_color_too_few_components(self):
        """parse_border_color should reject too few components."""
        with pytest.raises(ValueError):
            parse_border_color("255,255")

    def test_parse_invalid_color_too_many_components(self):
        """parse_border_color should reject too many components."""
        with pytest.raises(ValueError):
            parse_border_color("255,255,255,255")

    def test_parse_invalid_color_out_of_range(self):
        """parse_border_color should reject values out of range."""
        with pytest.raises(ValueError):
            parse_border_color("256,0,0")
        with pytest.raises(ValueError):
            parse_border_color("-1,0,0")

    def test_parse_invalid_color_non_numeric(self):
        """parse_border_color should reject non-numeric values."""
        with pytest.raises(ValueError):
            parse_border_color("red,green,blue")


class TestDefaultOutputPath:
    """Test default output path generation."""

    def test_default_output_png(self):
        """get_default_output_path should add _hexified suffix for png."""
        result = get_default_output_path(Path("/path/to/image.png"), "png")
        assert result == Path("/path/to/image_hexified.png")

    def test_default_output_jpg(self):
        """get_default_output_path should use correct format extension."""
        result = get_default_output_path(Path("/path/to/image.png"), "jpg")
        assert result == Path("/path/to/image_hexified.jpg")

    def test_default_output_preserves_directory(self):
        """get_default_output_path should preserve parent directory."""
        result = get_default_output_path(Path("/deep/nested/path/input.jpg"), "png")
        assert result.parent == Path("/deep/nested/path")

    def test_default_output_handles_dots_in_name(self):
        """get_default_output_path should handle files with dots in name."""
        result = get_default_output_path(Path("image.v2.final.png"), "png")
        assert result == Path("image.v2.final_hexified.png")


class TestImageDiscovery:
    """Test image file discovery for batch mode."""

    @pytest.fixture
    def temp_dir_with_images(self, tmp_path):
        """Create a temporary directory with test images."""
        # Create some test files
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.jpg").touch()
        (tmp_path / "image3.jpeg").touch()
        (tmp_path / "document.txt").touch()
        (tmp_path / "data.json").touch()

        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.png").touch()
        (subdir / "nested.bmp").touch()

        return tmp_path

    def test_discover_single_file(self, temp_dir_with_images):
        """discover_images should return single file if valid image."""
        image_path = temp_dir_with_images / "image1.png"
        result = discover_images(image_path)
        assert result == [image_path]

    def test_discover_single_file_non_image(self, temp_dir_with_images):
        """discover_images should return empty for non-image files."""
        text_path = temp_dir_with_images / "document.txt"
        result = discover_images(text_path)
        assert result == []

    def test_discover_directory_non_recursive(self, temp_dir_with_images):
        """discover_images should find images in directory (non-recursive)."""
        result = discover_images(temp_dir_with_images, recursive=False)
        assert len(result) == 3  # image1.png, image2.jpg, image3.jpeg

    def test_discover_directory_recursive(self, temp_dir_with_images):
        """discover_images should find images in subdirectories when recursive."""
        result = discover_images(temp_dir_with_images, recursive=True)
        assert len(result) == 5  # 3 in root + 2 in subdir

    def test_discover_with_pattern(self, temp_dir_with_images):
        """discover_images should filter by pattern."""
        result = discover_images(temp_dir_with_images, pattern="*.png")
        assert len(result) == 1
        assert result[0].name == "image1.png"

    def test_discover_with_pattern_recursive(self, temp_dir_with_images):
        """discover_images should apply pattern recursively."""
        result = discover_images(temp_dir_with_images, recursive=True, pattern="*.png")
        assert len(result) == 2  # image1.png and nested.png

    def test_discover_returns_sorted(self, temp_dir_with_images):
        """discover_images should return sorted list."""
        result = discover_images(temp_dir_with_images)
        assert result == sorted(result)


class TestPresetLoading:
    """Test preset configuration loading."""

    def test_all_presets_exist(self):
        """All expected presets should be defined."""
        expected = {"default", "fast", "detailed", "minimal"}
        assert set(PRESETS.keys()) == expected

    def test_preset_has_required_keys(self):
        """Each preset should have the required configuration keys."""
        for name, config in PRESETS.items():
            assert "num_palette_colors" in config, f"Preset {name} missing num_palette_colors"

    def test_load_settings_with_preset(self, tmp_path):
        """load_settings should apply preset values."""
        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="fast",
            config=None,
            colors=None,
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        settings = load_settings(args, logger)
        assert settings.num_palette_colors == 8  # fast preset value
        assert settings.quantization_method == QuantizationMethod.MINIBATCH_KMEANS

    def test_load_settings_cli_overrides_preset(self, tmp_path):
        """CLI arguments should override preset values."""
        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="fast",
            config=None,
            colors=24,  # Override the preset's 8
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        settings = load_settings(args, logger)
        assert settings.num_palette_colors == 24  # CLI value wins

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_settings_from_yaml(self, tmp_path):
        """load_settings should load from YAML file."""
        # Create a test YAML config
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
num_palette_colors: 20
num_layers: 6
border_width: 3
""")

        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="default",
            config=config_path,
            colors=None,
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        settings = load_settings(args, logger)
        assert settings.num_palette_colors == 20
        assert settings.num_layers == 6
        assert settings.border_width == 3

    def test_load_settings_from_json(self, tmp_path):
        """load_settings should load from JSON file."""
        import json

        config_path = tmp_path / "test_config.json"
        with open(config_path, "w") as f:
            json.dump({"num_palette_colors": 18, "num_zones": 15}, f)

        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="default",
            config=config_path,
            colors=None,
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        settings = load_settings(args, logger)
        assert settings.num_palette_colors == 18
        assert settings.num_zones == 15

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_settings_cli_overrides_config(self, tmp_path):
        """CLI arguments should override config file values."""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("num_palette_colors: 20")

        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="default",
            config=config_path,
            colors=30,  # Override config file's 20
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        settings = load_settings(args, logger)
        assert settings.num_palette_colors == 30  # CLI wins

    def test_load_settings_missing_config_file(self, tmp_path):
        """load_settings should raise error for missing config file."""
        logger = setup_logging(quiet=True)

        args = argparse.Namespace(
            preset="default",
            config=Path("/nonexistent/config.yaml"),
            colors=None,
            layers=None,
            zones=None,
            chunk_size=None,
            border_width=None,
            border_color=None,
            orientation=None,
        )

        with pytest.raises(FileNotFoundError):
            load_settings(args, logger)


class TestLogging:
    """Test logging setup."""

    def test_setup_logging_default(self):
        """setup_logging should create INFO level logger by default."""
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_setup_logging_verbose(self):
        """setup_logging with verbose=True should create DEBUG level logger."""
        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_setup_logging_quiet(self):
        """setup_logging with quiet=True should create ERROR level logger."""
        logger = setup_logging(quiet=True)
        assert logger.level == logging.ERROR


class TestMainFunction:
    """Test the main CLI entry point."""

    def test_main_list_presets(self, capsys):
        """main --list-presets should print presets and exit."""
        result = main(["input.png", "--list-presets"])
        assert result == 0

        captured = capsys.readouterr()
        assert "default" in captured.out
        assert "fast" in captured.out
        assert "detailed" in captured.out
        assert "minimal" in captured.out

    def test_main_input_not_found(self):
        """main should return error for nonexistent input."""
        result = main(["/nonexistent/image.png"])
        assert result == 1

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_main_save_config(self, tmp_path):
        """main --save-config should save settings and exit."""
        config_path = tmp_path / "saved_config.yaml"

        # Need a valid "input" even though we're just saving config
        # Use a real fixture path
        result = main([
            "dummy.png",  # Won't be processed
            "-c", "24",
            "--layers", "6",
            "--save-config", str(config_path),
        ])

        assert result == 0
        assert config_path.exists()

        # Verify saved settings
        settings = HexifySettings.from_yaml(str(config_path))
        assert settings.num_palette_colors == 24
        assert settings.num_layers == 6


class TestImageExtensions:
    """Test supported image extensions."""

    def test_common_extensions_supported(self):
        """Common image extensions should be supported."""
        expected = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        assert expected.issubset(IMAGE_EXTENSIONS)

    def test_extensions_are_lowercase(self):
        """All extensions should be lowercase."""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")


class TestPrintPresets:
    """Test preset printing."""

    def test_print_presets_output(self, capsys):
        """print_presets should output preset information."""
        print_presets()

        captured = capsys.readouterr()
        assert "Available Presets" in captured.out
        for preset_name in PRESETS:
            assert preset_name in captured.out
