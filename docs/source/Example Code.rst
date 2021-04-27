Example Code
===============================================
This example code demonstrates how to read, preprocess, train, and detect the speed using our library. 

.. code-block:: python

    import car_speed.car_speed_detection as car_speed

    car_speed.read('train.mp4', 'Car_Detection_images/')
    car_speed.preprocess('Car_Detection_images', 'train.txt', 'feature.txt', resize = 0.5, x_slice = 8, y_slice = 6)
    mse, MEAN_CONST, STD_CONST = car_speed.train('feature.txt')
    car_speed.speed_detection('Model.h5', 'test.mp4', 'detect_result.txt', 0.5, 8, 6, MEAN_CONST, STD_CONST)
    car_speed.combine_video_and_speed('test.mp4', 'detect_result.txt', 'test_with_speed.mp4')  

Line by line explanation
################################################

.. code-block:: python

    import car_speed.car_speed_detection as car_speed

Import the the car_speed_detection module from car_speed library as car_speed.

.. code-block:: python

    car_speed.read('train.mp4', 'Car_Detection_images/')

Read the training video from 'train.mp4' and store each frame into the 'Car_Detection_images/' directory.

.. code-block:: python

    car_speed.preprocess('Car_Detection_images', 'train.txt', 'feature.txt', resize = 0.5, x_slice = 8, y_slice = 6)

Read in all the frames two by two, resize them to 0.5 of the original size, calculate the optical flow between two frames, slice the output into 8 by 6 matrix, and flatten it into 1*48 array. If there are n frames, the output feature.txt would be a size n*48 matrix.

.. code-block:: python

    mse, MEAN_CONST, STD_CONST = car_speed.train('feature.txt')

Read in the feature set that was output from the preprocess function and train it with our Artificial Neural Network. The output would be a file named 'Model.h5'. Note that the function output MEAN_CONST and STD_CONST will be needed for speed_detection to normalized its data.

.. code-block:: python

    car_speed.speed_detection('Model.h5', 'test.mp4', 'detect_result.txt', 0.5, 8, 6, MEAN_CONST, STD_CONST)

Use the model 'Model.h5' to detect the speed of the car in the video 'test.mp4' and output the speed result to 'detect_result.txt'. Note that resize argument (0.5), x_slice argument (8), and y_slice argument (6), MEAN_CONST, and STD_CONST should be the same as the arguments that are used for preprocessing the model 'Model.h5'.

.. code-block:: python

    car_speed.combine_video_and_speed('test.mp4', 'detect_result.txt', 'test_with_speed.mp4') 

Print the speed of each frame on the video so it would be easier to visualize the result.

