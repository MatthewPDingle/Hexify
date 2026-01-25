# Hexify Example Configurations

This directory contains example configuration files for the Hexify CLI.

## Available Presets

### fast.yaml
Optimized for speed with reduced quality. Good for:
- Quick previews
- Batch processing large collections
- Testing different input images

```bash
hexify input.png --config examples/fast.yaml
```

### detailed.yaml
Maximum quality settings for detailed output. Best for:
- Final renders
- High-resolution artwork
- Images where color accuracy matters

```bash
hexify input.png --config examples/detailed.yaml
```

### minimal.yaml
Ultra-simplified style for artistic effects. Creates:
- Bold, graphic patterns
- Poster-like effects with limited colors
- Fast processing with distinctive look

```bash
hexify input.png --config examples/minimal.yaml
```

### bordered.yaml
Adds visible borders between hexagons for a stained-glass effect:
- Mosaic-like appearance
- Clear hexagon boundaries
- Artistic window effect

```bash
hexify input.png --config examples/bordered.yaml
```

## Creating Custom Configurations

You can create your own configuration by copying one of the examples:

```bash
cp examples/detailed.yaml my-config.yaml
# Edit my-config.yaml with your preferred settings
hexify input.png --config my-config.yaml
```

Or save your current CLI settings to a file:

```bash
hexify input.png -c 24 --layers 6 --save-config my-config.yaml
```

## Configuration Options

### Geometry Settings
- `hex_width`: Width of each hexagon in output pixels (default: 256)
- `hex_scale_factor`: Output scale relative to input (default: 16)
- `num_layers`: Number of concentric layers per hexagon (1-7)
- `num_zones`: Number of angular zones in even layers (4-24)
- `orientation`: Either `flat_top` or `pointy_top`

### Color Settings
- `color_space`: Color space for matching (`rgb`, `lab`, `hsv`)
- `quantization_method`: `kmeans` (accurate) or `minibatch_kmeans` (fast)
- `num_palette_colors`: Colors in the palette (2-64, recommended: 8-32)

### Style Settings
- `border_width`: Hexagon border width in pixels (0 = no border)
- `border_color`: RGB values as `[R, G, B]`

### Processing Settings
- `chunk_size`: Hexagons per processing batch (affects memory/speed)
- `max_palette_sample_pixels`: Max pixels sampled for palette generation

## Tips

1. **For portraits**: Use `detailed.yaml` with 24-32 colors for best results
2. **For landscapes**: `detailed.yaml` with 16-24 colors works well
3. **For icons/logos**: Try `minimal.yaml` for a graphic poster effect
4. **For batch processing**: Use `fast.yaml` to preview, then `detailed.yaml` for finals
