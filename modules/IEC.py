import os
import cv2
import numpy as np

def interpolate_borders(img, border_size=2):

    # Create a mask for the border region
    mask = np.zeros_like(img)
    mask[border_size:-border_size, border_size:-border_size] = 1

    # Create a context region around the border
    context_region = img[border_size:-border_size, border_size:-border_size]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the context region
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    context_clahe = clahe.apply(context_region)

    # Interpolate the border pixels using bicubic interpolation
    interpolated_img = img.copy()
    interpolated_img[border_size:-border_size, border_size:-border_size] = context_clahe

    return interpolated_img

def process_images_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                # Read the image in grayscale mode
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, 0)

                # Interpolate the borders using larger context and bicubic interpolation
                img_interpolated = interpolate_borders(img, border_size=2)

                # Save the processed image (overwrite the original image)
                cv2.imwrite(img_path, img_interpolated)

if __name__ == "__main__":
    folder_path = "../data/iu_xray/images"  # Specify the path to your image folder
    process_images_in_folder(folder_path)
