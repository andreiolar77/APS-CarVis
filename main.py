import cv2      # For video/image processing
from moviepy.editor import VideoFileClip    # For image recording and saving

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lane_detection_algorithm import process_frame
import json
import numpy as np

def video_process(path_in, path_out, mode, resolution):
    if mode == "read":
        video = cv2.VideoCapture(path_in) # Read video with opecv
        while True:            
            ret, frame = video.read()   # Read each frame of the video
            
            # Change the resolution
            if ret:     # Process while video is open
                if resolution == (1280, 720):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (1024, 768):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (800, 600):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (640, 480):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (400, 300):
                    resized = cv2.resize(frame, resolution)
                else:
                    raise Exception('No RESOLUTION IS SET!')

                final_frame = process_frame(resized)    # Process each frames
                cv2.imshow("Result", final_frame) # Output processed frames

                if cv2.waitKey(1) == 27:    # Break if ESC is pressed
                    break
            else:
                break   # Break when video is over.

        video.release() # Release frame
        cv2.destroyAllWindows() # Destroy all windows
    
    elif mode == "write":
        video_clip = VideoFileClip(path_in)   # Read image with moviepy
        project_video = video_clip.fl_image(process_frame)    # Process each frame of the video
        project_video.write_videofile(path_out, audio=False)    # Save whole video
    
    else:
        raise Exception("CHOOSE THE VIDEO MODE!")

if __name__ == "__main__":

    import os

    path = "VIL100/JPEGImages"

    # Get all items in the directory
    items = os.listdir(path)

    # Filter out non-directories
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

    folders.sort()
    # # Print the folder names
    # for folder in folders:
    #     print(folder)

    path2 = 'VIL100/Json'

    images = os.listdir(os.path.join(path, folders[0]))
    images.sort()

    acc = []

    for image in images:
        # print(image)

        image_path = os.path.join(path, folders[0], image)
        img = mpimg.imread(image_path)
        # plt.imshow(img)
        # plt.show()

        final_frame, lanes = process_frame(img)

        ground = []
        prediction = []

        # print(lanes)
        for line in lanes:
            for x1, y1, x2, y2 in line: # Unpack line by coordinates
                # fit a line to the points
                m = (y2 - y1) / (x2 - x1) # Slope
                b = y1 - m * x1 # y-intercept
                ground.append([m, b])
        # print(ground)


        with open(os.path.join(path2, folders[0], image) + '.json', 'r') as file:
            data = json.load(file)

        # Close the file
        file.close()

        lanes = data['annotations']['lane']
        mask = np.zeros_like(final_frame)   # Create array with zeros using the same dimension as frame
        for lane in lanes:
            if lane['lane_id'] == 1:
                points = (lane['points'])
                left1 = min(points, key=lambda x: x[0])
                left2 = max(points, key=lambda x: x[0])
                left1 = [int(i) for i in left1]
                left2 = [int(i) for i in left2]
                lx1, ly1 = left1
                lx2, ly2 = left2       
                cv2.line(mask, (lx1, ly1), (lx2, ly2), (255, 0, 0), 5)    # Draw the line on the created mask

            elif lane['lane_id'] == 2:
                points = (lane['points'])
                right1 = max(points, key=lambda x: x[0])
                right2 = min(points, key=lambda x: x[0])
                right1 = [int(i) for i in right1]
                right2 = [int(i) for i in right2]
                rx1, ry1 = right1
                rx2, ry2 = right2                      
                cv2.line(mask, (rx1, ry1), (rx2, ry2), (255, 0, 0), 5)    # Draw the line on the created mask

        lm = (ly2 - ly1) / (lx2 - lx1) # Slope
        lb = ly1 - lm * lx1 # y-intercept
        prediction.append([lm, lb])

        rm = (ry2 - ry1) / (rx2 - rx1) # Srope
        rb = ry1 - rm * rx1 # y-intercept
        prediction.append([rm, rb])

        # Access the data
        # print(data['annotations']['lane'])
        print_frame = cv2.addWeighted(final_frame, 0.8, mask, 1, 1)

        # print(ground, prediction)
        # print(prediction, ground)
        accuracy = 100 - (np.abs(np.divide(np.subtract(prediction,ground), ground)).mean()*100)
        acc.append(accuracy)
        # print(f'accuracy: {accuracy}%')
        # print(np.square(np.subtract(ground,prediction)).mean()*100)
        # exit()

        # calculate MSE between ground and prediction coresponding values
        # MSE = np.square(np.subtract(ground,prediction)).mean()
        # print(MSE)

        # plt.imshow(print_frame)
        # plt.title(image_path)
        # plt.show()
        # automatically close the plot after 1 second
        # plt.pause(0.01)
        
    print(acc)
    # make a list from 0 to len(acc)
    # from acc remove all values that are less than 0
    acc = [i for i in acc if i > 0]
    x = [i for i in range(len(acc))]
    # plot acc vs x
    plt.scatter(x, acc)
    plt.xlabel('frame')
    plt.ylabel('accuracy')
    plt.show()
    # video_process("input/white_road.mp4", "output/output_lane_detection.mp4", "write", (1280, 720))