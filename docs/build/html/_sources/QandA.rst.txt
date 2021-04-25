Q&A
===============================================
In this section, we provide answers to the common questions people have asked about this library.

Q1: What is the purpose of using this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	This library provides a camera-based solution that allows developers to use their own data or the data we provide to train the model, and detect the speed of the car using the preprocessing technique and ANN we propose.

Q2: Why should I choose your library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Our library is aiming to increase the accuracy and reduce the latency so that it could be run on all kinds of machines and not limited only to computers that have strong GPU, suitable for most cars these days. We are able to achieve MSE of 2 with 45 times fewer parameters used than the `research done by UCLA <https://ucladatares.medium.com/predicting-speed-from-video-frames-dissecting-the-comma-ai-challenge-5da697b55886>`_

Q3: How do I use this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We provide example code and API to help developers to understand how to use our library. See `Example code <https://car-speed-detection.readthedocs.io/en/latest/Example%20Code.html>`_ and `API <https://car-speed-detection.readthedocs.io/en/latest/API.html>`_ to know more.

Q4: What is optical flow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Optical flow is an algorithm that allows us to capture the moving object between two frames and transfer it into a velocity matrix. With optical flow, we could extract the information we need and get rid of the noises before training it with ANN. 

Q5: Why using the ANN instead of other networks to train the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	In our solution, the feature set we acquire from the preprocessing function has already contained the spatial relationship in each parameter, so we don't need to use the CNN to capture the spatial relationship again. The benefit of using ANN rather than CNN is that we could substantially reduce the parameters needed to train the model, which could reduce the latency greatly.

Q6: Is this library used for detecting other cars' speeds or the car I am driving?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	This library detects the speed of the car you are driving based on the video stream from your dashboard camera.

Q7: Do you provide data for training the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Yes, we provide 150 minutes of `training video <https://car-speed-detection.readthedocs.io/en/latest/Data.html>`_ generated from the car simulator `Carla <https://carla.org/>`_ that allows people to use it to train the model based on their operational domain. See more about `this Github repository developed by Christopher Crocker <https://github.com/CrockerC/carla_recording.git>`_ to know more about how we generated the data.

Q8: What library or package are used in this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We provide `Software Bill of Material <https://github.com/ShaoChiehLien/Car-Speed-Detection/blob/main/car_speed/bom.json>`_ using `CycloneDX <https://github.com/CycloneDX/cyclonedx-python>`_ that helps developers to understand our software architecture and could base on their need to decide whether they want to use our library.

Q9: What is the Software Bill of Material(SOB)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Software Bill of Material could trace down to the root library of all libraries that are included in our solution. Many users have their preference or security concerns about the certain library. They could use the SOB we provided to make sure it could satisfy their requirement. SOB also provides Hash Code for integrity checks for each library that's included in our solution.

Q10: Is Software Bill of Material(SOB) provided in this project?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Yes.

Q11: What license is provided for this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We choose the MIT license since we want this library to be open-sourced and can be accessed by everyone.

Q12: More Question?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Feel free to contact Shao-Chieh Lien at lienshaochieh@gmail.com if you have any more questions!

	