# Car-Speed-Detection
Car-Speed-Detection provides a python library to detect the speed of the driving 
car itself by the video stream from the dashboard camera installed on the car.

Car-Speed-Detection separates the speed detection process into three steps, 
preprocessing, training, and speed detection. By using Gunnar-Farneback optical flow
algorithm along with the pipeline we developed, we are able to extract each frame into
a small size matrix depends on developers preference. We use the Artifitial Neural 
Network (ANN) to train our model with the preprocessed matrix acquired from preprocessing
function. Developers could use the trained model to detect the speed of the car
at each frame using our speed detection function.

## Getting Started
### Installation
Car-Speed-Detection is available on [PyPI](https://pypi.org/project/car-speed-detection/) and can be
installed via [`pip`](https://pypi.org/project/pip/). See
[car-speed-detection.readthedocs.io](https://car-speed-detection.readthedocs.io/en/latest/)
to learn about system dependencies and installation alternatives and
recommendations.

```shell
pip install car-speed-detection
```

### Read, Preprocess, Train, and Detect the Car Speed
The [Car-Speed-Detection](https://pypi.org/project/car-speed-detection/) library consists of the
following parts:
- Read (Read the mp4 video and output each frame into a designated directory)
- Preprocess (Preprocess each frame and output a feature set for training)
- Train (Train the model using the feature set and Artifitial Neural Network)
- Speed Detection (Detect the speed using the model and video)

Take a look at the [API](https://car-speed-detection.readthedocs.io/en/latest/API.html#) to know more about
the Application Programming Interface and Sample for further information on how to use our library

## Result

## Acknowledge
Team Leader and Software Architecture: Shao-Chieh Lien 

Research Paper Writer: Meenakshi Pavithran 

Training Data Generator: Christopher Crocker 
