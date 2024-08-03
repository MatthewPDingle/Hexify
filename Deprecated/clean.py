from PIL import Image
import sys
import os

# Increase the MAX_IMAGE_PIXELS limit
Image.MAX_IMAGE_PIXELS = None

def is_surrounded_by_white_and_non_black(image, x, y):
    width, height = image.size
    white_pixels = 0
    non_black_pixels = 0
    white_pixel = (255, 255, 255)
    black_pixel = (0, 0, 0)
    
    # Check the 8 surrounding pixels
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                pixel = image.getpixel((nx, ny))
                if pixel == white_pixel:
                    white_pixels += 1
                elif pixel != black_pixel:
                    non_black_pixels += 1

    return (white_pixels >= 4) and ((white_pixels + non_black_pixels) >= 6)

def process_image_pass(image, pixels, width, height, total_pixels):
    processed_pixels = 0

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            processed_pixels += 1
            if processed_pixels % 1000 == 0:
                progress = (processed_pixels / total_pixels) * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}%")
                sys.stdout.flush()

            if pixels[x, y] == (0, 0, 0) and is_surrounded_by_white_and_non_black(image, x, y):
                pixels[x, y] = (255, 255, 255)
    print("\nPass completed.")

def process_image(input_path):
    # Open the input image
    image = Image.open(input_path)
    image = image.convert('RGB')  # Ensure image is in RGB mode
    pixels = image.load()
    width, height = image.size
    total_pixels = width * height

    # Perform the first pass
    print("Starting pass 1 of 1")
    process_image_pass(image, pixels, width, height, total_pixels)

    # Save the modified image with a "_clean" suffix
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_clean{ext}"
    image.save(output_path)
    print(f"\nProcessing completed and output saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    process_image(input_path)
