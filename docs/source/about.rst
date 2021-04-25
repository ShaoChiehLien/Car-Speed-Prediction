About
===============================================
Car-Speed-Detection provides a python library to detect the speed of the driving car itself by the video stream from the dashboard camera installed on the car.

Car-Speed-Detection separates the speed detection process into three steps, preprocessing, training, and speed detection. By using Gunnar-Farneback optical flow algorithm along with the pipeline we developed, we are able to extract each frame into a small size matrix depends on developers preference. We use the Artifitial Neural Network (ANN) to train our model with the preprocessed matrix acquired from preprocessing function. Developers could use the trained model to detect the speed of the car at each frame using our speed detection function.