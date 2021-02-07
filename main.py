import cv2
import time
import os
import re
import shutil


# read_path and write_path all must exist
# this function will clear out all files in write path before writing to it
# return how many jpg have been processed
def preprocess(read_path, write_dir):
    # Check if read_path and write_dir exist
    if not os.path.exists(read_path):
        print(f"read path: '{read_path}' doesn't exist")
        return False
    if not os.path.exists(write_dir):
        print(f"write directory: '{write_dir}' doesn't exist")
        return False
    # Check if write_dir is a directory
    if not os.path.isdir(write_dir):
        print(f"write directory: '{write_dir}' is not a directory")
        return False
    # Check if input format is mp4
    if not re.search(r'.*\.mp4', read_path):
        print(f"please input a mp4 file")
        return False

    # Start preprocess the video
    # Clear out the content in that directory first
    shutil.rmtree(write_dir)
    os.makedirs(write_dir)
    # Make sure the write_dir end with '/', so later could add index for each jpg file
    if write_dir[-1] != '/':
        write_dir += '/'

    cap = cv2.VideoCapture(read_path)  # read in the video
    index = 0
    while cap.isOpened():  # read the video frame by frame
        ret, frame = cap.read()
        if not ret:
            break
        # write jpg file in write_dir
        cv2.imwrite(write_dir + str(index) + '.jpg', frame)
        index += 1
        print(index)
    cap.release()
    cv2.destroyAllWindows()

    # return how many jpg have been processed
    return index


if __name__ == '__main__':
    print(preprocess('Data/train.mp4', 'Data/Car_Detection_images/'))

