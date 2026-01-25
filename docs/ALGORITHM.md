# Hexify Image Processing Algorithm

## Table of Contents
1. [Overview](#overview)
2. [Input/Output](#inputoutput)
3. [Hexagonal Grid System](#hexagonal-grid-system)
4. [7-Layer Concentric System](#7-layer-concentric-system)
5. [Color Selection](#color-selection)
6. [Zone Angle Calculation](#zone-angle-calculation)
7. [Mathematical Formulas](#mathematical-formulas)

---

## Overview

Hexify is an image processing algorithm that transforms input images into a stylized mosaic composed of hexagonal cells. Each hexagonal cell contains a sophisticated 7-layer concentric pattern that represents the average color of the corresponding region in the source image.

The algorithm achieves a unique aesthetic by:
- Dividing the output into a tessellated hexagonal grid
- Creating 7 concentric hexagonal layers within each cell
- Using alternating black/white and colored layers
- Implementing a 12-zone pie pattern for colored layers
- Intelligently selecting colors from a generated palette

```
Input Image                    Output Image
+-------------+               +---------------------------+
|             |               |   /\    /\    /\    /\   |
|  Original   |  =========>   |  /  \  /  \  /  \  /  \  |
|   Photo     |   Hexify      | /hex \/ hex\/ hex\/ hex\ |
|             |               | \ 7L /\ 7L /\ 7L /\ 7L / |
+-------------+               |  \  /  \  /  \  /  \  /  |
                              +---------------------------+
```

---

## Input/Output

### Input
- **Image**: Any standard image (RGB format)
- **Palette Size**: Number of colors to extract (default: 16)
- **Chunk Size**: Processing batch size (default: 32)

### Output
- **Scaled Image**: 16x the resolution of the input
- **Format**: RGB image with hexagonal cell patterns

### Scaling Factor

```
Output Resolution = Input Resolution x 16

Example:
  Input:  1920 x 1080 pixels
  Output: 30720 x 17280 pixels
```

### Hexagon Dimensions

| Property | Value | Formula |
|----------|-------|---------|
| Hex Width | 256 px | Fixed |
| Hex Height | 222 px | `256 * (sqrt(3)/2)` |
| Hex Radius | 128 px | `256 / 2` |
| Horizontal Spacing | 192 px | `256 * 0.75` |
| Vertical Spacing | 222 px | Same as height |

---

## Hexagonal Grid System

### Grid Layout

The hexagonal grid uses an offset coordinate system where odd columns are shifted vertically by half the hex height:

```
Column:    0       1       2       3       4
        +-----+ +-----+ +-----+ +-----+ +-----+
Row 0   | 0,0 | |     | | 0,2 | |     | | 0,4 |
        +-----+ |     | +-----+ |     | +-----+
              +-----+       +-----+
Row 1         | 1,1 |       | 1,3 |
              +-----+       +-----+
        +-----+       +-----+       +-----+
Row 1   | 1,0 |       | 1,2 |       | 1,4 |
        +-----+       +-----+       +-----+
```

### Center Calculation

For each hexagon at grid position (row, col):

```python
center_x = hex_horizontal_spacing * col
center_y = hex_vertical_spacing * row + (0.5 * hex_vertical_spacing if col % 2 else 0)
```

### Hexagon Mask Creation

Hexagons are oriented with a flat top (orientation = pi/2):

```
        ____
       /    \
      /      \
      \      /
       \____/
```

The mask is created using `RegularPolygon` with 6 vertices and the specified radius.

---

## 7-Layer Concentric System

Each hexagonal cell contains 7 concentric hexagonal layers, numbered from inside (1) to outside (7):

```
Cross-section view of a hexagonal cell:

         Layer 7 (outermost) - Black/White
        +---------------------------+
        |  Layer 6 - Colored (12 zones)
        | +----------------------+  |
        | | Layer 5 - Black/White|  |
        | | +------------------+ |  |
        | | | Layer 4 - Colored| |  |
        | | | +--------------+ | |  |
        | | | | L3 - B/W     | | |  |
        | | | | +----------+ | | |  |
        | | | | | L2 Color | | | |  |
        | | | | | +------+ | | | |  |
        | | | | | | L1   | | | | |  |
        | | | | | | B/W  | | | | |  |
        | | | | | +------+ | | | |  |
        | | | | +----------+ | | |  |
        | | | +--------------+ | |  |
        | | +------------------+ |  |
        | +----------------------+  |
        +---------------------------+
```

### Layer Types

| Layer | Type | Description |
|-------|------|-------------|
| 7 | Odd | Outermost, solid black or white |
| 6 | Even | Colored 12-zone pie pattern |
| 5 | Odd | Solid black or white |
| 4 | Even | Colored 12-zone pie pattern |
| 3 | Odd | Solid black or white |
| 2 | Even | Colored 12-zone pie pattern |
| 1 | Odd | Innermost, solid black or white |

### Odd Layers (1, 3, 5, 7) - Black/White

Odd layers are filled with either pure black (0, 0, 0) or pure white (255, 255, 255) based on the average brightness of the cell:

```python
brightness = mean(avg_rgb)
bw_color = (0, 0, 0) if brightness < 128 else (255, 255, 255)
```

**Layer Radii (Fixed Proportions)**:
```python
layer_radius[i] = outer_radius * (i / 7)

Layer 7: radius * (7/7) = 100% of hex radius
Layer 5: radius * (5/7) = ~71% of hex radius
Layer 3: radius * (3/7) = ~43% of hex radius
Layer 1: radius * (1/7) = ~14% of hex radius
```

### Even Layers (2, 4, 6) - Colored Pie Patterns

Even layers contain a 12-zone pie pattern with two alternating colors. The layer diameter varies based on brightness:

**Layer 6 Diameter**:
```python
diameter = 256 - abs(brightness - 128) * (256 - 192) / 128

When brightness = 128 (mid-gray): diameter = 256 (maximum)
When brightness = 0 or 255:       diameter = 192 (minimum)
```

**Layers 4 and 2 Diameter**:
```python
diameter = (layer_radii[i + 1] * 2) - abs(brightness - 128) * 64 / 128
```

This creates a visual effect where:
- Mid-brightness cells have larger colored layers
- Very dark or very bright cells have smaller colored layers

---

## Color Selection

### Palette Generation

1. **Downscale** large images to max 1 million pixels for efficiency
2. **K-Means Clustering** extracts the dominant colors:
   ```python
   kmeans = KMeans(n_clusters=num_palette_colors, random_state=42)
   palette = kmeans.cluster_centers_
   ```
3. **Sort** palette by brightness for consistency
4. **Round** and **hash** for caching purposes

### Color 1 Selection (Primary Color)

The closest palette color to the average RGB of the layer region:

```python
color_1 = closest_palette_color(avg_rgb, palette, avoid_rgb)
```

Colors used in previous layers are avoided to increase variety.

### Color 2 Selection (Secondary Color)

Color 2 is selected to minimize the difference between the weighted average of all colors in the layer and the target average RGB:

```python
mixed_rgb = (color_1_area * color_1 + color_2_area * color_2 + bw_area * bw_color) / total_area
color_2 = argmin(|avg_rgb - mixed_rgb|)
```

This ensures that when you "squint" at the output, the colors blend to approximate the original image.

---

## Zone Angle Calculation

### 12-Zone Pie Pattern

Each even layer is divided into 12 alternating zones radiating from the center:

```
              Zone 1
         _____|_____
        /     |     \
   Z12 /   Z2 | Z2   \ Z2
      /  _____|_____  \
     / /      |      \ \
 Z11|  | Center|       |  Z3
    |  |______|_______|  |
 Z10 \ \      |      / / Z4
      \ \_____|_____/ /
   Z9  \  Z8 | Z8   / Z5
        \____|____/
             Z7
             Z6
```

### Even vs Odd Zone Angles

The angular width of zones alternates:
- **Even zones** (2, 4, 6, 8, 10, 12): Use `even_angle`
- **Odd zones** (1, 3, 5, 7, 9, 11): Use `odd_angle`

```python
# Total must equal 360 degrees
6 * even_angle + 6 * odd_angle = 360
# Therefore:
even_angle + odd_angle = 60 degrees
```

### Calculating Zone Angles

The angle split is determined by how close Color 1 is to the target:

```python
# Distance from Color 1 to target
dist_1 = |color_1 - avg_rgb|

# Distance to nearest adjacent palette color
color_adj = closest_color_to(color_1) in palette
dist_adj = |color_1 - color_adj|

# Percentage determines angle split
percentage_off = min(dist_1, dist_adj) / max(dist_1, dist_adj)

even_angle = percentage_off * 60  # Range: 0 to 60 degrees
odd_angle = 60 - even_angle        # Range: 60 to 0 degrees
```

**Effect**:
- When Color 1 is an exact match: `even_angle = 0`, Color 1 zones are minimal
- When Color 1 is far from target: `even_angle = 60`, Color 1 zones are maximal

### Zone Vertex Calculation

For each zone, three key points define the triangular shape:

```
         p2
         /\
        /  \
       / p3 \
      /______\
  p1  center

Where:
- center: (radius, radius) - the hexagon center
- p1: Starting point on hexagon edge
- p2: Ending point on hexagon edge
- p3: Apex point for odd zones (creates extra triangle fill)
```

**Calculating p1 and p2**:
```python
start_angle = angle_offset
end_angle = start_angle + zone_angle

p1 = (center + hex_radius * cos(start_angle),
      center + hex_radius * sin(start_angle))

p2 = (center + hex_radius * cos(end_angle),
      center + hex_radius * sin(end_angle))
```

**Calculating p3 (Apex for Odd Zones)**:
```python
# Midpoint of p1-p2
p12 = ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

# Half-distance between p1 and p2
d = distance(p1, p2) / 2

# Length of shorter leg (30-60-90 triangle geometry)
length_of_shorter_leg = d / sqrt(3)

# Perpendicular offset from midpoint
dx = (p12.x - p1.x) / distance(p1, p12)
dy = (p12.y - p1.y) / distance(p1, p12)
p3 = (p12.x + length_of_shorter_leg * dy,
      p12.y - length_of_shorter_leg * dx)
```

The initial angle offset ensures zones align properly with the hexagon edges:
```python
angle_offset = 30 - (even_angle / 2)
```

---

## Mathematical Formulas

### Hexagon Area Formula

For a regular hexagon with radius `r` (center to vertex):

```
A = (3 * sqrt(3) / 2) * r^2

Derivation:
- A hexagon consists of 6 equilateral triangles
- Each triangle has side length r
- Area of equilateral triangle = (sqrt(3) / 4) * r^2
- Total area = 6 * (sqrt(3) / 4) * r^2 = (3 * sqrt(3) / 2) * r^2
```

**In code**:
```python
hex_area = 3 * math.sqrt(3) * (hex_radius ** 2) / 2
```

### Layer Diameter Calculations

**Layer 6 (Outermost Colored Layer)**:
```
diameter = 256 - |brightness - 128| * (256 - 192) / 128
         = 256 - |brightness - 128| * 64 / 128
         = 256 - |brightness - 128| / 2

Range: [192, 256] pixels
Maximum when brightness = 128
Minimum when brightness = 0 or 255
```

**Layers 4 and 2**:
```
diameter = 2 * layer_radii[i+1] - |brightness - 128| * 64 / 128

Where layer_radii[i+1] is the radius of the next outer odd layer
```

### Zone Area Calculations

For a pie slice (circular sector approximation):

```
zone_area = 0.5 * r^2 * theta

Where:
- r = layer radius
- theta = zone angle in radians
```

**Total Even Zone Area** (6 zones):
```python
even_area = 6 * (0.5 * hex_radius^2 * radians(even_angle))
```

**Total Odd Zone Area** (6 zones):
```python
odd_area = 6 * (0.5 * hex_radius^2 * radians(odd_angle))
```

### Color Mixing Weighted Average

The weighted average formula for selecting Color 2:

```
mixed_rgb = (A1 * C1 + A2 * C2 + Abw * Cbw) / (A1 + A2 + Abw)

Where:
- A1  = Total area of Color 1 zones (even zones)
- C1  = Color 1 RGB values
- A2  = Total area of Color 2 zones (odd zones)
- C2  = Color 2 RGB values (candidate)
- Abw = Area of black/white layer above
- Cbw = Black or white RGB (0,0,0 or 255,255,255)
```

**Goal**: Find C2 that minimizes:
```
|avg_rgb - mixed_rgb|
```

This Euclidean distance minimization ensures the blended appearance matches the original.

### Euclidean Distance (Color Matching)

```
distance = sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
```

**In code**:
```python
distance = np.linalg.norm(color_1 - color_2)
```

---

## Implementation Notes

### Caching System

Hexagon patterns are cached by their average RGB values to avoid redundant computation:

```python
cache_key = (int(avg_r), int(avg_g), int(avg_b))
```

This significantly improves performance for images with many similar-colored regions.

### Parallel Processing

The algorithm processes hexagons in chunks using `ProcessPoolExecutor`:
- Hexagons are grouped into chunks (default size: 32)
- Each chunk is processed by a separate worker process
- Results are composited onto the output image

### Point Clipping

Zone vertices that fall outside the hexagon boundary are clipped to the nearest edge:

```python
def clip_point_to_hexagon(point, hex_coords):
    if point is inside hexagon:
        return point
    else:
        return nearest_point_on_hexagon_edge(point)
```

This ensures all zone triangles render correctly within the hexagonal cell boundary.

---

## Quick Reference

| Constant | Value | Description |
|----------|-------|-------------|
| Scale Factor | 16x | Output resolution multiplier |
| Hex Width | 256 px | Base hexagon width |
| Hex Height | ~222 px | Based on sqrt(3)/2 ratio |
| Layers | 7 | Concentric hexagon layers |
| Zones per Layer | 12 | Pie slices in colored layers |
| Brightness Threshold | 128 | Black vs white determination |

---

## Pseudocode Summary

```
function hexify(input_image):
    output = create_blank_image(input.size * 16)
    palette = kmeans_extract_colors(input_image, n=16)

    for each hexagon_center in grid:
        avg_rgb = sample_input_region(hexagon_center / 16)
        bw_color = BLACK if mean(avg_rgb) < 128 else WHITE

        # Draw layers from outside to inside
        for layer in [7, 6, 5, 4, 3, 2, 1]:
            if layer is odd:
                draw_solid_hexagon(layer, bw_color)
            else:
                color_1 = closest_palette_color(avg_rgb)
                color_2 = select_complementary_color(avg_rgb, color_1)
                angles = calculate_zone_angles(avg_rgb, color_1)
                draw_12_zone_pattern(layer, color_1, color_2, angles)

        composite(output, hexagon_pattern)

    return output
```
