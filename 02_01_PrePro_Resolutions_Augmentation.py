from PIL import Image
import os
import time
import multiprocessing

# Path to the folder containing cropped images
cropped_folder = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Resolution\Train_2160p\Bad"

# Define the target resolutions
resolutions = {"1080p": (1920, 1080),
               "720p": (1280, 720),
               "480p": (854, 480),
               "360p": (640, 360)}

# Create folders for each resolution if it does't exist
output_folder = r"C:\ThesisMedia\CroppedPralines\Grouped_dataset\Thesis_Dataset_Resolution"
# for resolution in resolutions:
#     folder_path = os.path.join(output_folder, resolution)
#     os.makedirs(folder_path, exist_ok=True)

# Function to resize an image to a target resolution
def resize_image(image_path, output_path, reference_resolution):
    image = Image.open(image_path)
    original_width, original_height = image.size

    new_width = int((reference_resolution[0] / 3840) * original_width)
    new_height = int((reference_resolution[1] / 2160) * original_height)

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    resized_image.save(output_path)
    image.close()

if __name__ == "__main__":
    # Create a pool of worker processes
    pool = multiprocessing.Pool()
    start_time = time.time()

    # Iterate over the cropped images
    for filename in os.listdir(cropped_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(cropped_folder, filename)
            
            # Resize the image to each target resolution using multiprocessing
            for resolution, size in resolutions.items():
                output_path = os.path.join(output_folder, f'Train_{resolution}', "Bad",filename)
                pool.apply_async(resize_image, args=(image_path, output_path, size))

    # Close the pool and wait for all processes to finish
    end_time = time.time()
    print("Finished. Elapsed time : ", end_time-start_time)

    pool.close()
    pool.join()