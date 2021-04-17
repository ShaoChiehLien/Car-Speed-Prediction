import cv2
import time
import os
import re
import shutil
import math
import numpy as np
import pandas as pd
import tensorflow.keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


# read_path and write_path all must exist
# this function will clear out all files in write path before writing to it
# return how many jpg files have been write
def read(read_path, write_dir):
    """
        Load a video and stored each frame as a jpg file in a new directory.
    
        Args:
            read_path (str): Path to the input video, should be a mp4 file
            param (str): Path to the output directory
        Returns:
            int: The index of the last frames being processed
    """
    # Check if read_path and write_dir exist
    if not os.path.exists(read_path):
        print(f"read path: '{read_path}' doesn't exist")
        return False
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
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
        if index % 1000 == 0:
            print(index)
    cap.release()
    cv2.destroyAllWindows()

    # return how many jpg have been read
    return index


# Check if images in the directory match image_count
def check_images_match(read_dir, image_count):
    """
        Check if the images in the directory matches the image count
    
        Args:
            read_path (str): Path to the directory that stored the images
            image_count (int): Image count users expected
        Returns:
            bool: True if matches, False otherwise.
    """
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


def _slice_matrix(mag_matrix, x_slice, y_slice):
    # Calculate the sum of each mag area and return the sqrt of the area sum
    height, width = mag_matrix.shape
    height_seg_len = height // y_slice
    width_seg_len = width // x_slice
    result = []
    for h in range(y_slice):
        for w in range(x_slice):
            mag_area_sum = np.sum(
                mag_matrix[h * height_seg_len:(h + 1) * height_seg_len, w * width_seg_len:(w + 1) * width_seg_len])
            result.append(round(math.sqrt(mag_area_sum), 2))  # round the sqrt to 2

    return result


# _calculate_optical_mag will convert the images to grayscale before calculating the optical flow
def _calculate_optical_mag(image1, image2):
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

    return mag_matrix


# Preprocess all the images in read_dir, output a .txt file(output_path) containing optical flow matrix
# for each frame with corresponding speed
def preprocess(read_dir, train_path, output_path, resize=0.5, x_slice=8, y_slice=6):
    """
        Preprocess all the images and attach each frame with the speed and output a feature.txt file
    
        Args:
            read_dir (str): Path to the directory that stored the images
            train_path (str): Path to the file that store the speed of each frame in the video
            resize (int): Ratio factor along horizontal and vertical axis
            x_slice (int): Desired slice along horizontal axis
            y_slice (int): Desired slice along vertical axis
        Returns:
            float: time it takes to preprocess the images
    """
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
    if not check_images_match(read_dir, len(training_list)):
        print(f"images in directory {read_dir} doesn't match the length of training list")
        return False

    # Write the first row, from 0 to partX * partY, and 'weight'
    f = open(output_path, 'w')
    cols = [str(i) for i in range(x_slice * y_slice)]
    cols.append('weight\n')
    cols_str = ', '.join(str(i) for i in cols)
    f.write(cols_str)

    # Reference:https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    # read in the first image and resize it
    if read_dir[-1] != '/':
        read_dir += '/'
    image1 = cv2.imread(read_dir + '0' + '.jpg')
    height, width, _ = image1.shape
    image1 = cv2.resize(image1, (int(width * resize), int(height * resize)))

    for index in range(1, len(training_list)):
        # read in image2 and resize
        image2 = cv2.imread(read_dir + str(index) + '.jpg')
        height, width, _ = image2.shape
        image2 = cv2.resize(image2, (int(width * resize), int(height * resize)))

        # calculate optical flow and slice
        mag_matrix = _calculate_optical_mag(image1, image2)
        optical_mag_list = _slice_matrix(mag_matrix, x_slice, y_slice)

        # print magnitude matrix of current two matrix images
        optical_mag_string = ', '.join(str(i) for i in optical_mag_list)
        f.write(optical_mag_string)  # write the magnitude of the optical flow
        f.write(', ' + training_list[index - 1] + '\n')  # write the magnitude of the optical flow

        if index == len(training_list) - 1:
            f.write(optical_mag_string)  # write the magnitude of the optical flow
            f.write(', ' + training_list[index] + '\n')  # write the magnitude of the optical flow

        if index % 1000 == 0:  # print out the index every 1000 images
            print(index)
        image1 = image2
    f.close()

    print("Preprocessed Time: ", time.time() - start)
    return time.time() - start


def _get_dataset(read_path, split, shuf=True):
    if split > 1 or split < 0:
        print("split parameter out of range (0, 1)")
        return False, False, False, False
    if not os.path.exists(read_path):
        print(f"read path: '{read_path}' doesn't exist")
        return False, False, False, False

    # Read in csv file and shuffle the data if needed
    np.random.seed(10)
    reader = pd.read_csv(read_path)
    dataset = reader.values
    if shuf:
        np.random.shuffle(dataset)
    row, column = dataset.shape
    column -= 1

    # standardlize the data
    X = dataset[:, 0:column]
    Y = dataset[:, column]
    MEAN_CONST = X.mean(axis=0)
    STD_CONST = X.std(axis=0)
    X -= MEAN_CONST
    X /= STD_CONST
    # Assign training and testing data
    split_line = int(row * split)
    X_train, Y_train = X[:split_line], Y[:split_line]
    X_test, Y_test = X[split_line:], Y[split_line:]
    return X_train, Y_train, X_test, Y_test, MEAN_CONST, STD_CONST


def train(read_path, validation_split=0.75, batch_size=128, epoch=100, verbose=1):
    """
        Train the Artifitial Neural Network (ANN) model with the preprocessed feature set
    
        Args:
            read_path (str): Path to the file that store the feature set
            validation_split (float): Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch.
            batch_size (str): Integer or None. Number of samples per gradient update.
            epoch (int): Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
            verbose (int): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Note that the progress bar is not particularly useful when logged to a file, so verbose=2 is recommended when not running interactively
            
        Returns:
            tuple: tuple containing:
                (mse (float): mean square error of the model that is tested with validation data, MEAN_CONST (float): mean value that is used to normalize the feature set, STD_CONST (float): standard deviation value that is used to normalize the feature set)
    """
    # Check if read_path and write_dir exist
    if not os.path.exists(read_path):
        print(f"read path: '{read_path}' doesn't exist")
        return False
    X_train, Y_train, X_test, Y_test, MEAN_CONST, STD_CONST = _get_dataset(read_path, validation_split, shuf=True)

    # Build a Sequential Model
    model = Sequential([
        Dense(units=512, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=512, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(1)
    ])
    print(model.summary())

    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, verbose=verbose)
    mse, mae = model.evaluate(X_test, Y_test)
    print("MSE: %.2f" % mse)
    print("MAE_test: ", mae)
    model.save("Model.h5")

    return mse, MEAN_CONST, STD_CONST


# read video and output frame by frame
def speed_detection(model_path, video, output_path, required_resize, required_x_slice, required_y_slice, MEAN_CONST, STD_CONST):
    """
        Detect the speed of the automobile using the pretrained model and input video
    
        Args:
            model_path (str): Path to the pretrained model
            video (str): Path to the video
            output_path (str): Path to the output
            required_resize (int): Resize scale that was used for the pretrained model
            required_x_slice: x slice that was used for the pretrained model
            required_y_slice: y slice that was used for the pretrained model
            MEAN_CONST: Mean of the training set, used to normalize the testing set
            STD_CONST: Standard Deviation of the training set, used to normalize the testing set
            
        Returns:
            tuple: tuple containing:
                (index (int): index of the last frame that is predicted, detection_time (float): Total time took to predict the speed of the car from the input video)
    """
    start = time.time()  # start counting the speed_detection
    # Check if model and video exist
    if not os.path.exists(model_path):
        print(f"model path: '{model_path}' doesn't exist")
        return False
    if not os.path.exists(video):
        print(f"video: '{video}' doesn't exist")
        return False
    # Check if input format is .h5 and .mp4
    if not re.search(r'.*\.h5', model_path):
        print(f"please input a h5 file for model")
        return False
    if not re.search(r'.*\.mp4', video):
        print(f"please input a mp4 file for video")
        return False
    model = tensorflow.keras.models.load_model(model_path)  # load the model

    f = open(output_path, 'w')
    cap = cv2.VideoCapture(video)  # read in the video
    ret, image1 = cap.read()
    height, width, _ = image1.shape
    image1 = cv2.resize(image1, (int(width * required_resize), int(height * required_resize)))

    index = 0
    while cap.isOpened():  # read the video frame by frame
        ret, image2 = cap.read()
        if not ret:
            break
        # read in image2 and resize
        height, width, _ = image2.shape
        image2 = cv2.resize(image2, (int(width * required_resize), int(height * required_resize)))
        # calculate optical flow and slice
        mag_matrix = _calculate_optical_mag(image1, image2)
        optical_mag_list = _slice_matrix(mag_matrix, required_x_slice, required_y_slice)
        # print(optical_mag_list)
        images_to_predict = np.array(optical_mag_list)
        images_to_predict = images_to_predict.reshape(1, -1)
        images_to_predict -= MEAN_CONST
        images_to_predict /= STD_CONST
        # flatten the matrix so it could be input to the CNN model
        predictions = model.predict(x=images_to_predict)[0][0]
        f.write(str(predictions) + '\n')
        print(predictions)
        index += 1
        print(index)
        image1 = image2

    f.write(str(predictions))  # copy the last one again cause the frame difference
    index += 1
    f.close()
    cap.release()
    cv2.destroyAllWindows()

    # return how many speed for frame has been outputted
    detection_time = time.time() - start
    print("detection time: ", detection_time)
    print("image processed: ", index)
    return index, detection_time


def combine_video_and_speed(video_path, speed_path, output_path='combined_video_and_speed.mp4'):
    """
        Print the speed of the car on each frame in the video
    
        Args:
            video_path (str): Path to the video
            speed_path (str): Path to the speed text
            output_path (str): Path to the output

        Returns:
            bool: True if combining success, False if combining fail
    """
    if not os.path.exists(video_path):
        print(f"video path: '{video_path}' doesn't exist")
        return False
    if not os.path.exists(speed_path):
        print(f"speed_path path: '{speed_path}' doesn't exist")
        return False
    if not re.search(r'.*\.mp4', video_path):
        print(f"please input a mp4 file")
        return False
    if not re.search(r'.*\.txt', speed_path):
        print(f"please input a txt file")
        return False

    # Get the frame count of the video
    cap = cv2.VideoCapture(video_path)

    # Get the speed list
    f = open(speed_path, 'r')
    speed_list = f.read().split('\n')
    f.close()

    # Check if length of speed list matches the total frame of the video
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != len(speed_list):
        print("Lenth of speed list doesn\'t match the total frame of the video")

    # Write
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v is for .mp4 file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height of video
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    frame_index = 0
    while cap.isOpened():  # show the video frame by frame
        ret, frame = cap.read()  # ret = 1 if cap is read and store in frame
        if ret:
            # set up font and text
            font = cv2.FONT_ITALIC
            text = f'Speed: {speed_list[frame_index]}'

            # add text to frame
            frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)  # write the original frame to out's frame

            frame_index += 1
            print(frame_index)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True


# Load in pretrained model and retrain it with more data
def fine_tune(read_model_path, read_feature_path, output_path, validation_split=0.75, batch_size=128, epoch=100, verbose=1):
    """
        Load in pretrained model and retrain it with more data.
    
        Args:
            read_model_path (str): Path to the pre-trained model
            read_feature_path (str): Path to the feature set
            output_path (str): Path to the output

        Returns:
            tuple: tuple containing:
                (mse (float): mean square error of the model that is tested with validation data, MEAN_CONST (float): mean value that is used to normalize the feature set, STD_CONST (float): standard deviation value that is used to normalize the feature set)
    """
    # Check if read_model_path and read_feature_path exist
    if not os.path.exists(read_model_path):
        print(f"read model path: '{read_model_path}' doesn't exist")
        return False
    if not os.path.exists(read_feature_path):
        print(f"read feature path: '{read_feature_path}' doesn't exist")
        return False
    X_train, Y_train, X_test, Y_test, MEAN_CONST, STD_CONST = _get_dataset(read_feature_path, validation_split, shuf=True)

    model = load_model(read_model_path)
    print("original model: ", model.summary())
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.fit(X_train, Y_train, epochs=epoch, batch_size=batch_size, verbose=verbose)
    mse, mae = model.evaluate(X_test, Y_test)
    print("MSE: %.2f" % mse)
    print("MAE_test: ", mae)
    model.save(output_path)
    print("fine tuned model: ", model.summary())
    return mse, MEAN_CONST, STD_CONST


if __name__ == '__main__':
    # read('Data/train.mp4', 'Data/Car_Detection_images/')
    # preprocess('Data/Car_Detection_images', 'train.txt', 'Data/feature.txt', resize = 0.5, x_slice = 8, y_slice = 6)
    # mse, MEAN_CONST, STD_CONST = train('Data/feature.txt')
    # speed_detection('Model.h5', 'Data/train_test.mp4', 'compare.txt', 0.5, 8, 6, MEAN_CONST, STD_CONST)
    # combine_video_and_speed('Data/test_trained_model/test.mp4', 'Data/test_trained_model/test_output.txt', 'Data/combined_video_and_speed.mp4')

    # fine_tune('Model.h5', 'Data/feature.txt', 'Model_tune.h5')
    pass
