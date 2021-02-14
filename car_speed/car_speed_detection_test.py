import car_speed_detection
import cv2
import os
import shutil
import numpy as np
import math
import pandas as pd
from sklearn.metrics import mean_squared_error


# Unit Testing: units in preprocess pipeline
# Test if read in the video and output each frame successfully
def test_read():
    # Create a new directory
    write_dir = 'test_case/images_dir/'
    if os.path.exists(write_dir):  # If exist, delete it
        shutil.rmtree(write_dir)
    os.makedirs(write_dir)  # create an empty file

    # Read the video and save each frame into the new created directory
    car_speed_detection.read('test_case/test.mp4', write_dir)
    # Calculate the total frame
    video = cv2.VideoCapture('test_case/test.mp4')
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Check if the read function's output match the total_frames
    for index in range(0, int(total_frames)-1):
        if not os.path.exists(write_dir + str(index) + '.jpg'):
            print('test_read: FAIL')
            return False

    # Delete the testing file
    shutil.rmtree(write_dir)

    # If not deleted, return false
    if os.path.exists(write_dir):
        print('test_read: FAIL')
        return False

    print("test_read: PASS")
    return True


# Unit Testing: units in preprocess pipeline
# Test if images match the count
def test_check_images_match():
    if not car_speed_detection.check_images_match('test_case/test_check_images_match', 120):
        print("test_check_images_match: FAIL")
        return False
    print("test_check_images_match: PASS")
    return True


# Unit Testing: units in preprocess pipeline
# Test if slice matrix works
def test_slice_matrix():
    test_arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]])
    sliced_matrix = car_speed_detection.slice_matrix(test_arr, 4, 3)
    ans_arr = [5.48, 8.94, 11.4, 13.42, 13.42, 11.4, 8.94, 5.48, 3.16, 4.47, 5.48, 6.32]
    if sliced_matrix != ans_arr:
        print("test_slice_matrix: FAIL")
        return False

    print("test_slice_matrix: PASS")
    return True


# Unit Testing: units in preprocess pipeline
# Use if the calculate_optical_mag result matches the result produced by the algorithm provided by openCV
def test_calculate_optical_mag():
    frame1_path = 'test_case/test_calculate_optical_mag/frame1.jpg'
    frame2_path = 'test_case/test_calculate_optical_mag/frame2.jpg'

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
    test_arr = car_speed_detection.calculate_optical_mag(cv2.imread(frame1_path), cv2.imread(frame2_path))
    if (test_arr != ans_arr).any():
        print("test_calculate_optical_mag: FAIL")
        return False
    if not ((ans_arr == test_arr).all()):
        print("test_calculate_optical_mag: FAIL")
        return False

    # Visualize the optical flow
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('test_case/test_calculate_optical_mag/output.png', bgr)

    print("test_calculate_optical_mag: PASS")
    return True


# Integration Testing: test preprocess pipeline
# Can't generate test case, so make sure each step follows the plan
def test_preprocess():
    # Generate answer data
    # Generate three preprocessed image matrix with corresponding speed
    ans_arr = []
    for i in range(2):
        # resize
        image1 = cv2.imread("test_case/test_preprocess/" + str(i) + '.jpg')
        image2 = cv2.imread("test_case/test_preprocess/" + str(i+1) + '.jpg')
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
    car_speed_detection.preprocess('test_case/test_preprocess', 'test_case/test_preprocess/train.txt',
                                   'test_case/test_preprocess/dummy', resize=0.5, x_slice=8, y_slice=6)
    reader = pd.read_csv('test_case/test_preprocess/dummy')
    test_arr = reader.values

    # Compare the testing data and answer
    if not (test_arr == ans_arr).all():
        print("Integration Testing-test preprocess pipeline: FAIL")
        os.remove('test_case/test_preprocess/dummy')
        return False

    print("Integration Testing-test preprocess pipeline: PASS")
    os.remove('test_case/test_preprocess/dummy')
    return True


# Unit Testing: units in training section
# Use manual created test case to test if read in the dataset successfully
def test_get_dataset():
    feature_path = 'test_case/test_get_dataset/feature.txt'
    test_X_tra, test_Y_tra, test_X_tes, test_Y_tes = car_speed_detection.get_dataset(feature_path, 0.5, shuf=False)
    ans_X_tra = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18],
                          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    ans_Y_tra = np.array([20, 100.2])
    # Check if the shape and the value of test_X_tra and test_Y_tra is correct
    if test_X_tra.shape != ans_X_tra.shape or test_Y_tra.shape != ans_Y_tra.shape:
        print("test_get_dataset: FAIL")
        return False
    if not ((test_X_tra == ans_X_tra).all() and (test_Y_tra == ans_Y_tra).all()):
        print("test_get_dataset: FAIL")
        return False

    ans_X_tes = np.array([[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
                          [10000, 9999, 9998, 9997, 9996, 9995, 9994, 9993, 9992],
                          [100, 99, 98, 97, 96, 95, 94, 93, 92]])
    ans_Y_tes = np.array([1, 0.222, 10000.00])
    # Check if the shape and the value of test_X and test_Y is correct
    if test_X_tes.shape != ans_X_tes.shape or test_Y_tes.shape != ans_Y_tes.shape:
        print("test_get_dataset: FAIL")
        return False
    if not ((test_X_tes == ans_X_tes).all() and (test_Y_tes == ans_Y_tes).all()):
        print("test_get_dataset: FAIL")
        return False

    print("test_get_dataset: PASS")
    return True


# Integration Testing: test training section
def test_train():
    error_list = []
    total_mse = 0.0
    for i in range(10):
        current_mse = car_speed_detection.train('Data/feature.txt')
        error_list.append(current_mse)
        total_mse += current_mse
        if current_mse > 15:  # If one model have mse higher than 15, testing fail
            print("test_train: FAIL")
            return False, error_list

    if total_mse/10 > 10:  # If average of 10 models have mse higher than 10, testing fail
        print("test_train: FAIL")
        return False, error_list

    print("test_train: PASS")
    return True, error_list


def test_plot_scatter():
    car_speed_detection.plot_scatter('test_case/test_speed_detection/train.txt',
                                     'test_case/test_speed_detection/test.txt')


def test_speed_detection():

    model_path = 'test_case/test_speed_detection/test_Model.h5'
    video = 'test_case/test_speed_detection/test.mp4'
    output_path = 'test_case/test_speed_detection/test.txt'
    car_speed_detection.speed_detection(model_path, video, output_path, 0.5, 8, 6)
    # read real data
    ans_path = 'test_case/test_speed_detection/answer.txt'
    f = open(ans_path, 'r')
    ans_list = f.read().split('\n')
    f.close()
    # read prediction data
    f = open(output_path, 'r')
    test_list = f.read().split('\n')
    f.close()

    # convert string list to float list
    for i in range(0, len(ans_list)):
        ans_list[i] = float(ans_list[i])
        test_list[i] = float(test_list[i])
    # check MSE error, if > 1.5, fail, suppose to be 1.012948989868164
    mse = mean_squared_error(ans_list, test_list)
    if mse >= 1.5:
        print("mse", mse)
        print("test_speed_detection: FAIL")
        return False
    print("test_speed_detection: PASS")
    # NEED to delete file !!!
    return True


def integration_testing():
    pass


if __name__ == '__main__':
    # print(test_read())
    # print(test_slice_matrix())
    # print(test_calculate_optical_mag())
    # print(test_get_dataset())
    # print(test_check_images_match())
    # print(test_preprocess())
    # test_train()
    test_speed_detection()
    # test_plot_scatter()
'''
    car_speed_detection.preprocess()
    car_speed_detection.train()
    car_speed_detection.plot_scatter()
    car_speed_detection.speed_detection()
'''
