"""
Tests for style augmentation features.

This module tests the style rendering utilities including:
- Border rendering (solid, double, glow)
- Gradient application (radial, linear)
- Noise texture application
- RGBA output with transparency
"""

import numpy as np
import pytest
import cv2

from v2.styles import (
    BorderStyle,
    FillStyle,
    add_hexagon_border,
    apply_radial_gradient,
    apply_linear_gradient,
    add_noise_texture,
    apply_alpha_channel,
    apply_vignette,
    get_style_preset,
    apply_style_preset,
    STYLE_EFFECTS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_pattern():
    """Create a sample RGB pattern for testing."""
    pattern = np.full((100, 100, 3), 128, dtype=np.uint8)
    return pattern


@pytest.fixture
def sample_rgba_pattern():
    """Create a sample RGBA pattern for testing."""
    pattern = np.full((100, 100, 4), 128, dtype=np.uint8)
    pattern[:, :, 3] = 255  # Fully opaque
    return pattern


@pytest.fixture
def hexagon_coords():
    """Create hexagon vertex coordinates for a centered hexagon."""
    import math
    center = (50, 50)
    radius = 40
    coords = []
    for i in range(6):
        angle = math.radians(60 * i + 30)  # Flat-top orientation
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        coords.append([x, y])
    return np.array(coords, dtype=np.int32)


@pytest.fixture
def hexagon_mask():
    """Create a hexagon mask for testing."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Create a simple circular approximation of hexagon for testing
    cv2.circle(mask, (50, 50), 40, 255, -1)
    return mask


# =============================================================================
# Border Rendering Tests
# =============================================================================

class TestBorderRendering:
    """Tests for hexagon border rendering."""

    def test_no_border_returns_unchanged(self, sample_pattern, hexagon_coords):
        """Border with NONE style should return unchanged pattern."""
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.NONE
        )
        np.testing.assert_array_equal(result, sample_pattern)

    def test_zero_width_returns_unchanged(self, sample_pattern, hexagon_coords):
        """Border with zero width should return unchanged pattern."""
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=0,
            border_style=BorderStyle.SOLID
        )
        np.testing.assert_array_equal(result, sample_pattern)

    def test_solid_border_modifies_pattern(self, sample_pattern, hexagon_coords):
        """Solid border should modify the pattern along edges."""
        border_color = (255, 0, 0)  # Red
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=2,
            border_color=border_color,
            border_style=BorderStyle.SOLID
        )
        # Pattern should be modified (not identical to original)
        assert not np.array_equal(result, sample_pattern)
        # Check that red color appears in result
        assert np.any(result[:, :, 0] == 255)

    def test_double_border_creates_two_lines(self, sample_pattern, hexagon_coords):
        """Double border should create inner and outer lines."""
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=4,
            border_color=(0, 0, 0),
            border_style=BorderStyle.DOUBLE
        )
        # Pattern should be modified
        assert not np.array_equal(result, sample_pattern)

    def test_glow_border_creates_fading_effect(self, sample_pattern, hexagon_coords):
        """Glow border should create multiple fading layers."""
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=3,
            border_color=(255, 255, 255),
            border_style=BorderStyle.GLOW
        )
        # Pattern should be modified
        assert not np.array_equal(result, sample_pattern)

    def test_border_preserves_shape(self, sample_pattern, hexagon_coords):
        """Border should preserve pattern dimensions."""
        result = add_hexagon_border(
            sample_pattern,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.SOLID
        )
        assert result.shape == sample_pattern.shape


# =============================================================================
# Gradient Application Tests
# =============================================================================

class TestGradientApplication:
    """Tests for gradient fill application."""

    def test_radial_gradient_basic(self, sample_pattern):
        """Radial gradient should create color variation from center."""
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=50,
            inner_color=(255, 0, 0),
            outer_color=(0, 0, 255)
        )
        # Center should be closer to inner_color (red)
        center_pixel = result[50, 50]
        assert center_pixel[0] > center_pixel[2]  # More red than blue

        # Edge should be closer to outer_color (blue)
        edge_pixel = result[0, 50]
        assert edge_pixel[2] > edge_pixel[0]  # More blue than red

    def test_radial_gradient_with_mask(self, sample_pattern, hexagon_mask):
        """Radial gradient with mask should only affect masked area."""
        original = sample_pattern.copy()
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=50,
            inner_color=(255, 0, 0),
            outer_color=(0, 0, 255),
            mask=hexagon_mask
        )
        # Outside mask should be unchanged (approximately - some edge effects)
        # Check a corner that's definitely outside the circle mask
        assert result[0, 0, 0] == original[0, 0, 0]

    def test_linear_gradient_basic(self, sample_pattern):
        """Linear gradient should create color transition along direction."""
        result = apply_linear_gradient(
            sample_pattern,
            start_point=(0, 0),
            end_point=(99, 99),
            start_color=(255, 0, 0),
            end_color=(0, 0, 255)
        )
        # Top-left should be closer to start_color (red)
        start_pixel = result[0, 0]
        assert start_pixel[0] > start_pixel[2]

        # Bottom-right should be closer to end_color (blue)
        end_pixel = result[99, 99]
        assert end_pixel[2] > end_pixel[0]

    def test_linear_gradient_with_mask(self, sample_pattern, hexagon_mask):
        """Linear gradient with mask should only affect masked area."""
        original = sample_pattern.copy()
        result = apply_linear_gradient(
            sample_pattern,
            start_point=(0, 0),
            end_point=(99, 99),
            start_color=(255, 0, 0),
            end_color=(0, 0, 255),
            mask=hexagon_mask
        )
        # Corner outside mask should be unchanged
        assert result[0, 0, 0] == original[0, 0, 0]

    def test_gradient_preserves_shape(self, sample_pattern):
        """Gradient should preserve pattern dimensions."""
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=50,
            inner_color=(255, 0, 0),
            outer_color=(0, 0, 255)
        )
        assert result.shape == sample_pattern.shape


# =============================================================================
# Noise Texture Tests
# =============================================================================

class TestNoiseTexture:
    """Tests for noise texture application."""

    def test_zero_intensity_returns_unchanged(self, sample_pattern):
        """Zero noise intensity should return unchanged pattern."""
        result = add_noise_texture(sample_pattern, intensity=0.0)
        np.testing.assert_array_equal(result, sample_pattern)

    def test_noise_modifies_pattern(self, sample_pattern):
        """Noise should modify the pattern values."""
        np.random.seed(42)  # For reproducibility
        result = add_noise_texture(sample_pattern, intensity=0.1)
        assert not np.array_equal(result, sample_pattern)

    def test_noise_within_valid_range(self, sample_pattern):
        """Noise should keep values within valid [0, 255] range."""
        np.random.seed(42)
        result = add_noise_texture(sample_pattern, intensity=0.5)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_noise_intensity_affects_variation(self, sample_pattern):
        """Higher intensity should create more variation."""
        np.random.seed(42)
        result_low = add_noise_texture(sample_pattern.copy(), intensity=0.01)
        np.random.seed(42)
        result_high = add_noise_texture(sample_pattern.copy(), intensity=0.2)

        # Calculate standard deviation of differences
        diff_low = np.std(result_low.astype(float) - sample_pattern.astype(float))
        diff_high = np.std(result_high.astype(float) - sample_pattern.astype(float))

        assert diff_high > diff_low

    def test_noise_with_mask(self, sample_pattern, hexagon_mask):
        """Noise with mask should only affect masked area."""
        np.random.seed(42)
        original = sample_pattern.copy()
        result = add_noise_texture(sample_pattern, intensity=0.1, mask=hexagon_mask)

        # Corner outside mask should be unchanged
        assert result[0, 0, 0] == original[0, 0, 0]

    def test_noise_preserves_shape(self, sample_pattern):
        """Noise should preserve pattern dimensions."""
        result = add_noise_texture(sample_pattern, intensity=0.1)
        assert result.shape == sample_pattern.shape


# =============================================================================
# RGBA/Alpha Channel Tests
# =============================================================================

class TestAlphaChannel:
    """Tests for RGBA output with transparency."""

    def test_rgba_output_has_four_channels(self, sample_pattern, hexagon_mask):
        """RGBA output should have 4 channels."""
        result = apply_alpha_channel(sample_pattern, hexagon_mask)
        assert result.shape[2] == 4

    def test_alpha_inside_mask_is_opaque(self, sample_pattern, hexagon_mask):
        """Pixels inside mask should be fully opaque (alpha=255)."""
        result = apply_alpha_channel(sample_pattern, hexagon_mask)
        # Check a pixel inside the mask (center)
        assert result[50, 50, 3] == 255

    def test_alpha_outside_mask_is_transparent(self, sample_pattern, hexagon_mask):
        """Pixels outside mask should be transparent (alpha=0 by default)."""
        result = apply_alpha_channel(sample_pattern, hexagon_mask, background_alpha=0)
        # Check a corner outside the mask
        assert result[0, 0, 3] == 0

    def test_custom_background_alpha(self, sample_pattern, hexagon_mask):
        """Custom background alpha should be applied outside mask."""
        result = apply_alpha_channel(sample_pattern, hexagon_mask, background_alpha=128)
        # Corner should have custom alpha
        assert result[0, 0, 3] == 128

    def test_rgb_values_preserved(self, sample_pattern, hexagon_mask):
        """RGB values should be preserved in RGBA output."""
        result = apply_alpha_channel(sample_pattern, hexagon_mask)
        # RGB channels should match original
        np.testing.assert_array_equal(result[:, :, :3], sample_pattern)


# =============================================================================
# Vignette Effect Tests
# =============================================================================

class TestVignetteEffect:
    """Tests for vignette effect application."""

    def test_zero_strength_returns_unchanged(self, sample_pattern):
        """Zero vignette strength should return unchanged pattern."""
        result = apply_vignette(sample_pattern, center=(50, 50), radius=50, strength=0)
        np.testing.assert_array_equal(result, sample_pattern)

    def test_vignette_darkens_edges(self, sample_pattern):
        """Vignette should darken pixels toward edges."""
        result = apply_vignette(sample_pattern, center=(50, 50), radius=50, strength=0.5)

        # Center should be brighter than edge
        center_brightness = np.mean(result[50, 50])
        edge_brightness = np.mean(result[0, 0])
        assert center_brightness >= edge_brightness


# =============================================================================
# Style Preset Tests
# =============================================================================

class TestStylePresets:
    """Tests for style effect presets."""

    def test_classic_preset_is_empty(self):
        """Classic preset should have no effects."""
        preset = get_style_preset("classic")
        assert preset == {}

    def test_outlined_preset_has_border(self):
        """Outlined preset should have border settings."""
        preset = get_style_preset("outlined")
        assert "border_width" in preset
        assert preset["border_width"] > 0

    def test_textured_preset_has_noise(self):
        """Textured preset should have noise settings."""
        preset = get_style_preset("textured")
        assert "noise_intensity" in preset
        assert preset["noise_intensity"] > 0

    def test_unknown_preset_raises_error(self):
        """Unknown preset name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_style_preset("unknown_preset")
        assert "unknown_preset" in str(exc_info.value)

    def test_all_presets_are_valid(self):
        """All defined presets should be retrievable."""
        for preset_name in STYLE_EFFECTS.keys():
            preset = get_style_preset(preset_name)
            assert isinstance(preset, dict)

    def test_apply_style_preset_classic(self, sample_pattern, hexagon_coords, hexagon_mask):
        """Applying classic preset should not modify pattern."""
        result = apply_style_preset(
            sample_pattern,
            hexagon_coords,
            center=(50, 50),
            radius=50,
            preset_name="classic"
        )
        np.testing.assert_array_equal(result, sample_pattern)

    def test_apply_style_preset_outlined(self, sample_pattern, hexagon_coords, hexagon_mask):
        """Applying outlined preset should add borders."""
        result = apply_style_preset(
            sample_pattern,
            hexagon_coords,
            center=(50, 50),
            radius=50,
            preset_name="outlined"
        )
        assert not np.array_equal(result, sample_pattern)


# =============================================================================
# Integration Tests
# =============================================================================

class TestStyleIntegration:
    """Integration tests for combining multiple style effects."""

    def test_combined_noise_and_border(self, sample_pattern, hexagon_coords):
        """Applying both noise and border should work together."""
        np.random.seed(42)
        # First apply noise
        result = add_noise_texture(sample_pattern, intensity=0.05)
        # Then apply border
        result = add_hexagon_border(
            result,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.SOLID
        )
        # Should be different from original
        assert not np.array_equal(result, sample_pattern)
        # Should have valid dimensions
        assert result.shape == sample_pattern.shape

    def test_gradient_then_border(self, sample_pattern, hexagon_coords):
        """Applying gradient then border should work correctly."""
        # Apply gradient
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=50,
            inner_color=(255, 128, 128),
            outer_color=(128, 128, 255)
        )
        # Apply border
        result = add_hexagon_border(
            result,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.SOLID
        )
        assert result.shape == sample_pattern.shape

    def test_all_effects_combined(self, sample_pattern, hexagon_coords, hexagon_mask):
        """All effects should work when combined."""
        np.random.seed(42)

        # Start with gradient
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=50,
            inner_color=(200, 200, 200),
            outer_color=(100, 100, 100)
        )

        # Add noise
        result = add_noise_texture(result, intensity=0.03, mask=hexagon_mask)

        # Add vignette
        result = apply_vignette(result, center=(50, 50), radius=50, strength=0.2)

        # Add border
        result = add_hexagon_border(
            result,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.SOLID
        )

        # Verify output
        assert result.shape == sample_pattern.shape
        assert result.dtype == np.uint8


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_pattern(self):
        """Small pattern should be handled correctly."""
        small_pattern = np.zeros((10, 10, 3), dtype=np.uint8)
        result = add_noise_texture(small_pattern, intensity=0.1)
        assert result.shape == small_pattern.shape

    def test_single_pixel_pattern(self):
        """Single pixel pattern should not crash."""
        tiny_pattern = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = apply_radial_gradient(
            tiny_pattern,
            center=(0, 0),
            radius=1,
            inner_color=(255, 0, 0),
            outer_color=(0, 0, 255)
        )
        assert result.shape == tiny_pattern.shape

    def test_zero_radius_gradient(self, sample_pattern):
        """Zero radius gradient should not crash."""
        result = apply_radial_gradient(
            sample_pattern,
            center=(50, 50),
            radius=0,
            inner_color=(255, 0, 0),
            outer_color=(0, 0, 255)
        )
        assert result.shape == sample_pattern.shape

    def test_rgba_pattern_with_border(self, sample_rgba_pattern, hexagon_coords):
        """RGBA patterns should work with border."""
        result = add_hexagon_border(
            sample_rgba_pattern,
            hexagon_coords,
            border_width=2,
            border_style=BorderStyle.SOLID
        )
        assert result.shape == sample_rgba_pattern.shape

    def test_high_intensity_noise_clamped(self, sample_pattern):
        """Noise with intensity > 1 should be clamped."""
        np.random.seed(42)
        result = add_noise_texture(sample_pattern, intensity=2.0)
        assert np.all(result >= 0)
        assert np.all(result <= 255)
