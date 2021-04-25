Q&A
===============================================
In this section, we provided answer to the questions people have asked about this library.

Q1: What is purpose of using this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	This library provides a camera-based solution that allow developer to train the model, and detect the speed of the car using the preprocess technique and ANN we proposed.

Q2: Why should I choose your library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Our library is aiming to increase the accuracy and reduce the latency so that it would be more suitable in the scenarios that doesn't have strong GPU, like most of the cars these days. We were able to achieve MSE of 2 with 45 times less parameters used than the `research done by UCLA <https://ucladatares.medium.com/predicting-speed-from-video-frames-dissecting-the-comma-ai-challenge-5da697b55886>`_

Q3: How do I use this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We provided example code and API to help developers to understand how to use our library. See `Example code <https://car-speed-detection.readthedocs.io/en/latest/Example%20Code.html>`_ and `API <https://car-speed-detection.readthedocs.io/en/latest/API.html>`_ to know more.

Q4: What is optical flow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Optical flow is an algorithm that allow us to capture the moving object between frames and transfer it into a velocity matrix. With optical flow, we could extract the information we need and get rid of the noises before training it with ANN. 

Q5: Why using ANN model instead of other networks for training model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	In our solution, the feature set we acquired from the preprocessing function has already contained the spatial relationship at each parameters, so we don't need to use the CNN to capture the spatial relationship again. The benefit of using ANN rather than CNN is that we could substantially reduce the parameters needed to train the model, which could save us lots of time.

Q6: Is this library used for detecting other car's speed or the car I am driving?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	This library detects the speed of the car you are driving based on the video stream from your dashboard camera.

Q7: Do you provide data for training the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Yes, we generated 150 minutes `training video <https://car-speed-detection.readthedocs.io/en/latest/Data.html`_ using the car simulator `Carla <https://carla.org/>`_ that allows people to use it to train based on their operational domain. See more about `this Github repository <https://github.com/CrockerC/carla_recording.git>`_ to know more about how we generated the data.

Q8: What library or package are used in this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We provides `Software Bill of Material <https://github.com/ShaoChiehLien/Car-Speed-Detection/blob/main/car_speed/bom.json>`_ using `CycloneDX <https://github.com/CycloneDX/cyclonedx-python>`_ that helps developers to understand our software architecture and could base on their need to decide whether they want to use our library.

Q9: What is Software Bill of Material(SOB)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Software Bill of Material could trace down to the root library of all libraries that are included in our library. Many users have their preference or security concern about certain library. They could use the SOB we provided to make sure it could satisfies their requirement. SOB also provides Hash Code for integrity check for each library that's included in our solution.

Q10: Is Software Bill of Material(SOB) provided in this project?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	Yes.

Q11: What license is provided for this library?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	We choose MIT license since we want this library to be open-sourced and can be accessed by everyone.

Q12: Future work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	