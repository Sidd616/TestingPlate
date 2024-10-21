from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def resize_image(image, new_width, new_height):
    # Create an empty image
    resized_image = Image.new("RGB", (new_width, new_height))

    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate width and height factors
    n_width_factor = orig_width / new_width
    n_height_factor = orig_height / new_height

    for x in range(new_width):
        for y in range(new_height):
            # Find the coordinates in the original image
            fr_x = int(x * n_width_factor)
            fr_y = int(y * n_height_factor)
            cx = min(fr_x + 1, orig_width - 1)
            cy = min(fr_y + 1, orig_height - 1)

            # Calculate the interpolation factors
            fx = x * n_width_factor - fr_x
            fy = y * n_height_factor - fr_y
            nx = 1.0 - fx
            ny = 1.0 - fy

            # Get the pixel colors
            color1 = np.array(image.getpixel((fr_x, fr_y)))
            color2 = np.array(image.getpixel((cx, fr_y)))
            color3 = np.array(image.getpixel((fr_x, cy)))
            color4 = np.array(image.getpixel((cx, cy)))

            # Interpolate colors
            blue = (nx * color1[2] + fx * color2[2]) * ny + (ny * (nx * color1[2] + fx * color2[2]) + fy * (nx * color3[2] + fx * color4[2]))
            green = (nx * color1[1] + fx * color2[1]) * ny + (ny * (nx * color1[1] + fx * color2[1]) + fy * (nx * color3[1] + fx * color4[1]))
            red = (nx * color1[0] + fx * color2[0]) * ny + (ny * (nx * color1[0] + fx * color2[0]) + fy * (nx * color3[0] + fx * color4[0]))

            resized_image.putpixel((x, y), (int(red), int(green), int(blue)))

    # Apply grayscale and noise removal
    resized_image = set_grayscale(resized_image)
    resized_image = remove_noise(resized_image)

    return resized_image

def set_grayscale(image):
    return image.convert("L").convert("RGB")

def remove_noise(image):
    width, height = image.size
    noisy_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if pixel[0] < 162 and pixel[1] < 162 and pixel[2] < 162:
                noisy_image.putpixel((x, y), (0, 0, 0))  # Black
            elif pixel[0] > 162 and pixel[1] > 162 and pixel[2] > 162:
                noisy_image.putpixel((x, y), (255, 255, 255))  # White
            else:
                noisy_image.putpixel((x, y), pixel)  # Retain original color

    return noisy_image

def save_with_dpi(image, filename, dpi=(300, 300)):
    image.save(filename, dpi=dpi)

def main(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Resize parameters
    new_width, new_height = 2000, 2000  # You can adjust the size as needed

    # Process the image
    processed_image = resize_image(img, new_width, new_height)
    
     # Save the processed image with 300 DPI
    save_with_dpi(processed_image, 'processed_image_300dpi.jpg')

    # Show the original and processed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_image)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Change 'demo_image.jpg' to the path of your demo image
    main('demo.jpg')
