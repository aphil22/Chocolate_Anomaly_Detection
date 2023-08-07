import cv2
import os
import time
import numpy as np
import traceback
import multiprocessing


# Define co-ordinates for pixels to compare the values which are on the frame
# Define variables for the video
pixel_y1 = 200 #320  #210
pixel_x1 = 1080  #1327 #1030
pixel_y2 = 250 #369  #425  
pixel_x2 = 1080  #1327 #1030  

def read_color(input_image, pixel_x, pixel_y):
    pixel_value = input_image[pixel_y, pixel_x]
    return pixel_value

def hsv_mask_state(frame):
    lower = np.array([100,80,0])
    upper = np.array([179,255,150])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    pixel1 = read_color(mask, pixel_x1, pixel_y1)
    pixel2 = read_color(mask, pixel_x2, pixel_y2)
    # print(pixel1, pixel2)
    return (pixel1 == 255 and pixel2 == 0)

def video_to_frames(input_loc, output_loc, video_file):
    
    global cap
    global count

    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        ## Write the results back to output location.
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)   #rotate frame coz of my mistake
        state = hsv_mask_state(frame)
        print('reading frame: ', count)

        if state == True:
            cv2.imwrite(output_loc + "/4K-" + video_file +"-%#05d.jpg" % (count+1), frame)  ###change file index
            print('Writing frame: ', count)
        count = count + 50                           ########### change incriments here for frame
        cap.set(cv2.CAP_PROP_POS_FRAMES,count)
            # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for reading." % (time_end-time_start))
            break


if __name__ == "__main__":
    count = 0
    video_files = ["P", "R", "T", "V"]
    folder_path = r"C:\ThesisMedia\videos_gopro\GoPro\Videos/"

    processes = []
    for video_file in video_files:
        input_loc = folder_path + os.path.splitext(os.path.basename(video_file))[0]+".MP4"
        output_loc = r"C:\ThesisMedia\videos_gopro\GoPro\Cropped images\Extracted frame\4k_rot"
        process = multiprocessing.Process(target=video_to_frames, args=(input_loc, output_loc, video_file))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All videos processed.")