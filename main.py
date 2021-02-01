import cv2
import os


def read_video(video, path='data/image/'):

    cap = cv2.VideoCapture(video)

    if not os.path.exists(path):  # make a data directory if not exists
        os.makedirs(path)

    frame_count = 0
    while cap.isOpened():  # output each frame to file
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(path + str(frame_count) + '.jpg', frame)
            frame_count += 1
            print(frame_count)
            if frame_count >= 100:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def preprocess(path='data/image/'):  # preprocess the images

    if not os.path.exists(path):  # make a data directory if not exists
        print('Please specify a file to preprocess the images')

    if not os.path.exists(path):  # make a data directory if not exists
        os.makedirs(path)

    preprocessed_data_path = 'data/preprocessed_data_path/'
    frame_count = 0
    while True:
        img1_path = path + str(frame_count) + '.jpg'
        img2_path = path + str(frame_count+1) + '.jpg'
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = cv2.imread(img1_path, 0)  # read in 1st image in grayscale
            img2 = cv2.imread(img2_path, 0)  # read in 2nd image in grayscale
            diff_frame = img2 - img1

        else:
            # extend one previous image
            cv2.imwrite(preprocessed_data_path + str(frame_count) + '.jpg',
                        preprocessed_data_path + str(frame_count-1) + '.jpg')
            break

        frame_count += 1


read_video('train.mp4')
