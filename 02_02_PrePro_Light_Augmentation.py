import cv2
import os
import numpy as np
from multiprocessing import Pool

def increase_brightness(image, value):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV image into channels
    h, s, v = cv2.split(hsv)

    # Increase the value channel (brightness) by the specified value
    v = cv2.add(v, value)

    # Clip the values to the valid range of 0-255
    v = np.clip(v, 0, 255)

    # Merge the channels back together
    hsv = cv2.merge((h, s, v))

    # Convert the HSV image back to the BGR color space
    brighter_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return brighter_image

def process_image(image_path, output_folder_base, increment_values):
    # Load the original image
    image = cv2.imread(image_path)

    # Apply brightness modifications and save to respective output folders
    for value in increment_values:
        # Increase the brightness by the current value
        brighter_image = increase_brightness(image, value)

        # Save the modified image to the respective output folder
        folder_name = output_folder_base + str(value)
        output_path = os.path.join(folder_name, os.path.basename(image_path))
        cv2.imwrite(output_path, brighter_image)

    print(f"Processed: {image_path}")

def main():
    # Path to the folder containing the original images
    folder_path = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Lighting\Train_720p\Bad"

    # Output folder base name
    output_folder_base = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Lighting\Train_720p\Bad1\Light_"

    # Brightness increment values
    increment_values = [-90, -60, -30, 0, 30, 60, 90]

    # Create the output folders
    for value in increment_values:
        folder_name = output_folder_base + str(value)
        os.makedirs(folder_name, exist_ok=True)

    # Iterate over the files in the folder and process them using multiprocessing
    pool = Pool(processes=os.cpu_count())  # Use the maximum number of available CPUs
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
    pool.starmap(process_image, [(image_path, output_folder_base, increment_values) for image_path in image_paths])
    pool.close()
    pool.join()

    print("Brightness modification completed.")

if __name__ == '__main__':
    main()
