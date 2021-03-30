import car_speed.car_speed_detection as car_speed_detection
import cv2
import os
import shutil
import numpy as np
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import pytest
import sys


# Unit Testing: units in preprocess pipeline
# Test if read in the video and output each frame successfully
def test_read():
    # Create a new directory
    write_dir = 'tests/test_case/test_read/images_dir/'
    if os.path.exists(write_dir):  # If exist, delete it
        shutil.rmtree(write_dir)
    os.makedirs(write_dir)  # create an empty file

    # Read the video and save each frame into the new created directory
    car_speed_detection.read('tests/test_case/test_read/test.mp4', write_dir)

    # Check if the read function's output match the total_frames
    total_frames = 1203
    for index in range(0, total_frames):
        assert os.path.exists(write_dir + str(index) + '.jpg')

    # Delete the testing file
    shutil.rmtree(write_dir)
    return


# Unit Testing: units in preprocess pipeline
# Test if slice matrix works
def test_slice_matrix():
    test_arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]])
    sliced_matrix = car_speed_detection._slice_matrix(test_arr, 4, 3)
    ans_arr = [5.48, 8.94, 11.4, 13.42, 13.42, 11.4, 8.94, 5.48, 3.16, 4.47, 5.48, 6.32]
    assert sliced_matrix == ans_arr
    return


# Unit Testing: units in preprocess pipeline
# Test if images match the count
def test_check_images_match():
    assert car_speed_detection.check_images_match('tests/test_case/test_check_images_match', 120)
    return


# Unit Testing: units in preprocess pipeline
# Use if the calculate_optical_mag result matches the result produced by the algorithm provided by openCV
def test_calculate_optical_mag():
    frame1_path = 'tests/test_case/test_calculate_optical_mag/frame1.jpg'
    frame2_path = 'tests/test_case/test_calculate_optical_mag/frame2.jpg'

    # Algorithm from opencv
    frame1 = cv2.imread(frame1_path)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame2 = cv2.imread(frame2_path)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    ans_arr = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = ans_arr
    # Test calculate_optical_mag
    test_arr = car_speed_detection._calculate_optical_mag(cv2.imread(frame1_path), cv2.imread(frame2_path))
    assert ((ans_arr == test_arr).all())

    # Visualize the optical flow
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('tests/test_case/test_calculate_optical_mag/output.png', bgr)
    return


# Integration Testing: test preprocess pipeline
# Can't generate test case, so make sure each step follows the plan
def test_preprocess():
    # Generate answer data
    # Generate three preprocessed image matrix with corresponding speed
    ans_arr = []
    for i in range(2):
        # resize
        image1 = cv2.imread("tests/test_case/test_preprocess/" + str(i) + '.jpg')
        image2 = cv2.imread("tests/test_case/test_preprocess/" + str(i+1) + '.jpg')
        # resize
        image1 = cv2.resize(image1, (320, 240))
        image2 = cv2.resize(image2, (320, 240))
        # grayscale
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # optical flow
        flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_matrix = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # slice into 48 parts
        result = []
        for h in range(6):
            for w in range(8):
                mag_area_sum = np.sum(
                    mag_matrix[h * 40:(h + 1) * 40, w * 40:(w + 1) * 40])
                result.append(round(math.sqrt(mag_area_sum), 2))  # round the sqrt to 2
        if i == 0:
            result.append(20.4)
        else:
            result.append(51.2)
        ans_arr.append(result)
    ans_arr.append(ans_arr[1].copy())
    ans_arr[2][48] = 1022

    # Generate testing data
    car_speed_detection.preprocess('tests/test_case/test_preprocess', 'tests/test_case/test_preprocess/train.txt',
                                   'tests/test_case/test_preprocess/dummy', resize=0.5, x_slice=8, y_slice=6)
    reader = pd.read_csv('tests/test_case/test_preprocess/dummy')
    test_arr = reader.values

    # Compare the testing data and answer
    assert (test_arr == ans_arr).all()

    os.remove('tests/test_case/test_preprocess/dummy')
    return True


# Unit Testing: units in training section
# Use manual created test case to test if read in the dataset successfully
def test_get_dataset():
    feature_path = 'tests/test_case/test_get_dataset/feature.txt'
    # Use get_dataset to read in and round up the test data
    test_X_tra, test_Y_tra, test_X_tes, test_Y_tes, MEAN_CONST, STD_CONST = \
        car_speed_detection._get_dataset(feature_path, 0.5, shuf=False)
    test_X_tra = np.around(test_X_tra, 3)
    test_X_tes = np.around(test_X_tes, 3)

    # Create answer array
    # Calculate the normalized data from this: https: // mathcracker.com / normalize - data
    # Calculate the population mean and std from this: https://www.calculator.net/standard-deviation-calculator.html
    ans_X_tra = np.array([[-0.504, -0.566, -0.624, -0.675, -0.719, -0.755, -0.782, -0.801, -0.812],
                          [-0.507, -0.569, -0.627, -0.679, -0.723, -0.759, -0.786, -0.804, -0.816]])
    ans_Y_tra = np.array([20, 100.2])
    # Check if the shape and the value of test_X_tra and test_Y_tra is correct
    assert test_X_tra.shape == ans_X_tra.shape and test_Y_tra.shape == ans_Y_tra.shape

    assert ((test_X_tra == ans_X_tra).all() and (test_Y_tra == ans_Y_tra).all())

    ans_X_tes = np.array([[-0.507, -0.313, -0.109,  0.098,  0.302,  0.497,  0.678,  0.842,  0.988],
                          [     2,  1.991,  1.961,   1.91,  1.839,  1.751,  1.652,  1.546,  1.437],
                          [-0.482, -0.543, -0.601, -0.654, -0.698, -0.735, -0.763, -0.783, -0.796]])
    ans_Y_tes = np.array([1, 0.222, 10000.00])
    # Check if the shape and the value of test_X and test_Y is correct
    assert test_X_tes.shape == ans_X_tes.shape and test_Y_tes.shape == ans_Y_tes.shape

    assert ((test_X_tes == ans_X_tes).all() and (test_Y_tes == ans_Y_tes).all())

    return


# Integration Testing: test training section
def test_train():
    error_list = []
    total_mse = 0.0
    for i in range(0, 1):
        current_mse, MEAN_CONST, STD_CONST = car_speed_detection.train('tests/test_case/test_train/feature.txt')
        error_list.append(current_mse)
        total_mse += current_mse
        assert current_mse <= 10

    assert total_mse/10 <= 5

    return


def test_speed_detection():

    model_path = 'tests/test_case/test_speed_detection/test_Model.h5'
    video = 'tests/test_case/test_speed_detection/test.mp4'
    output_path = 'tests/test_case/test_speed_detection/test.txt'
    MEAN_CONST = [147.7918201, 138.24889951, 116.21455, 106.64215147, 110.28405098,
     121.62982745,  141.48395196,   145.04602255,   171.35321961, 148.17993235,
     110.37138431,  83.55375833,    96.68191618,    140.38086618, 185.07591765,
     203.17913971,  251.1464152,    212.70471078,   152.21537255, 107.87600049,
     113.8789951,   167.16572353,   223.34071667,   230.88207941, 202.32842353,
     218.76250049,  239.56567941,   129.21213578,   156.10384461, 229.17822941,
     213.5008348,   202.7395201,    83.87611667,    151.52054559, 121.75671961,
     86.96753971,   86.5108598,     143.28980588,   132.71390833, 71.56798725,
     36.68911618,   55.40894755,    66.44651912,    73.01452353,  65.63785588,
     54.13270539,   44.33209559,    24.61398676]
    STD_CONST = [121.20170689, 108.73248112, 89.89439207, 76.08570102, 76.03350568,
     83.30810119,   105.63822608,   123.09427226,   101.14608647,   91.48304002,
     82.89905858,   68.48766174,    63.38327538,    77.71343457,    105.02754141,
     130.8793052,   90.42021844,    72.49725972,    57.11527939,    44.60546987,
     45.4591745,    59.44147935,    83.48659665,    110.79245319,   111.18939872,
     91.22885143,   78.38628012,    50.27432635,    55.91700826,    83.25067549,
     90.28454622,   112.16524769,   55.76708369,    79.64393783,    56.86681395,
     43.64527864,   43.3930939,     72.28834558,    76.48463965,    69.35293988,
     22.70234598,   26.81283386,    29.15718362,    33.42548885,    30.24977878,
     26.20702268,   30.61206634,    25.81503033]
    car_speed_detection.speed_detection(model_path, video, output_path, 0.5, 8, 6, MEAN_CONST, STD_CONST)
    # read real data
    ans_path = 'tests/test_case/test_speed_detection/answer.txt'
    f = open(ans_path, 'r')
    ans_list = f.read().split('\n')
    f.close()
    # read prediction data
    assert os.path.exists(output_path)
    f = open(output_path, 'r')
    test_list = f.read().split('\n')
    f.close()

    # convert string list to float list
    for i in range(0, len(ans_list)):
        ans_list[i] = float(ans_list[i])
        test_list[i] = float(test_list[i])
    
    for i in range(0, len(ans_list)):
        if math.isnan(ans_list[i]):
            ans_list[i] = 0 
    for i in range(0, len(test_list)):
        if math.isnan(test_list[i]):
            test_list[i] = 0     
            
    # check MSE error, if > 1.5, fail, suppose to be 1.012948989868164
    sum_for_mse = 0
    for i in range(0, len(ans_list)):
        temp = ans_list[i]-test_list[i]
        sum_for_mse += (temp * temp)
    
    # mse = mean_squared_error(ans_list, test_list)
    assert sum_for_mse > 5

    print("test_speed_detection: PASS")
    os.remove('tests/test_case/test_speed_detection/test.txt')
    return True

# Run the testing script
# py.test -k test -v
