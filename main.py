import cv2
import time
import os
import re
import shutil
import math
import numpy as np


# read_path and write_path all must exist
# this function will clear out all files in write path before writing to it
# return how many jpg files have been write
def read(read_path, write_dir):
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

    # Start read the video
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

    # return how many jpg have been read
    return index


def check_dir_images(read_dir, image_count):
    # Check if read directory exists
    if not os.path.exists(read_dir):
        print(f"read directory: '{read_dir}' doesn't exist")
        return False
    # Check if read is a directory
    if not os.path.isdir(read_dir):
        print(f"read directory: '{read_dir}' is not a directory")
        return False

    # Make sure the read_dir end with '/', so later could add index for each jpg file
    if read_dir[-1] != '/':
        read_dir += '/'

    for index in range(0, image_count):
        if not os.path.exists(read_dir + str(index) + '.jpg'):
            return False
    return True


# calculate_optical_mag will convert the images to grayscale before calculating the optical flow
def calculate_optical_mag(image1, image2, x_slice, y_slice):
    # Check if image have the same size
    if image1.shape != image2.shape:
        print("Image has different size")
        return False

    # Convert image to grayscale to reduce noise
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow, the return vector is in Cartesian Coordinates
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract the magnitude of each vector by transforming Cartesian Coordinates to Polar Coordinates
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # normalize the magnitude
    mag_matrix = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Calculate the sum of each mag area and return the sqrt of the area sum
    height, width = mag_matrix.shape
    height_seg_len = height // y_slice
    width_seg_len = width // x_slice
    result = []
    for h in range(y_slice):
        for w in range(x_slice):
            mag_area_sum = np.sum(mag_matrix[h*height_seg_len:h*(height_seg_len+1), w*width_seg_len:w*(width_seg_len+1)])
            result.append(round(math.sqrt(mag_area_sum), 2)) # round the sqrt to 2
    return result


# Preprocess all the images in read_dir, output a .txt file(output_path) containing optical flow matrix
# for each frame with corresponding speed
def preprocess(read_dir, train_path, output_path, resize=0.5, x_slice=8, y_slice=6):
    start = time.time()  # start counting the preprocess time
    # Check if read directory exists
    if not os.path.exists(read_dir):
        print(f"read directory: '{read_dir}' doesn't exist")
        return False
    # Check if read is a directory
    if not os.path.isdir(read_dir):
        print(f"read directory: '{read_dir}' is not a directory")
        return False
    # Check if training path exists
    if not os.path.exists(train_path):
        print(f"training path: '{train_path}' doesn't exist")
        return False

    # Check if images in read_dir have same size as train_path
    f = open(train_path, 'r')
    training_list = f.read().split('\n')
    f.close()

    # Check if the images in directory match the length of the training list
    if not check_dir_images(read_dir, len(training_list)):
        print(f"images in directory {read_dir} doesn't match the length of training list")
        return False

    # Write the first row, from 0 to partX * partY, and 'weight'
    f = open(output_path, 'w')
    cols = [str(i) for i in range(x_slice * y_slice)]
    cols.append('weight\n')
    cols_str = '\t'.join(str(i) for i in cols)
    f.write(cols_str)

    # Referece:https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    # read in the first image and resize it
    if read_dir[-1] != '/':
        read_dir += '/'
    image1 = cv2.imread(read_dir + '0' + '.jpg')
    print(image1.shape)
    height, width, _ = image1.shape
    image1 = cv2.resize(image1, (int(width*resize), int(height*resize)))

    for index in range(1, len(training_list)):
        image2 = cv2.imread(read_dir + str(index) + '.jpg')
        height, width, _ = image2.shape
        image2 = cv2.resize(image2, (int(width * resize), int(height * resize)))

        optical_mag_list = calculate_optical_mag(image1, image2, x_slice, y_slice)
        optical_mag_string = '\t'.join(str(i) for i in optical_mag_list)
        f.write(optical_mag_string)  # write the magnitude of the optical flow
        f.write('\t' + training_list[index-1] + '\n')  # write the magnitude of the optical flow

        if index == len(training_list)-1:
            f.write(optical_mag_string)  # write the magnitude of the optical flow
            f.write('\t' + training_list[index] + '\n')  # write the magnitude of the optical flow

        image1 = image2
    f.close()

    print("Preprocessed Time: ", time.time()-start)
    return time.time()-start


if __name__ == '__main__':
    # print('File read:' + str(read('Data/train.mp4', 'Data/Car_Detection_images/')))
    # print(read('Data/train.mp4', 'Data/Car_Detection_images/'))
    # preprocess('Data/Car_Detection_images', 'Data/train copy.txt', 1, 2)
    preprocess('Data/Car_Detection_images', 'Data/train copy.txt', 'Data/test.txt')
