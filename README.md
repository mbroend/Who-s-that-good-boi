# Who-s-that-good-boi
Repository for final Udacity submission for the Data Science Nanodegree.

## Problem statement
Using machine learning models to classify images are one of the most profound uses of AI in modern times.
With libraries such as tensorflow and keras one can utilize the work of very brilliant people to create a (relatively) fast web application that can achieve something we all care about.
Namely to tell us the race of all the good doggos we're going to encounter in the world.

## Overview of algorithm
The algorithm has three objectives:
1 Classify human/no-human 
2 Clasisfy dog/no-dog
3 Classify dog race
This is achived using three seperate pretrained models with different strategies.
* To classify human/no-human the opencv haar-cascade algorithm is used. https://docs.opencv.org/trunk/db/d28/tutorial_cascade_classifier.html
* To classify dog/no-dog the ResNet50 CNN is used
* To classify the dog race transfer learning is used based on the InceptionResNetV2.
** The last layers are discared and a additional layers are added and trained on labeled dog images
** As the dataset is small the weights of the CNN is kept the same  and only ending layers are trained


## Packages and other recourses
* cv2 4.1.2 (pip install opencv-python)
* Flask 1.1.1
* Keras 2.3.1
* Tensorflow 1.14.0
* Numpy 1.16.2
* PIL 7.0.0
* tqdm 4.42.0
* Imagenet labels https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
* Dog dataset can be found here https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip and should be extracted as a subfolder to models


## Instructions
Serving the web application:
* cd to app folder
* serve the app by executin python app.py
* Open a browser and enter your local ip address
* Make sure port 5000 is allowed in your firewall
** Alternatively alter app.run(host='0.0.0.0', port=5000, debug=True) to app.run(host='127.0.0.1', port=5000, debug=True) in app.py
(Re)training the model:
* cd to models folder
* run model_train.py script

## Etc.
A note on doggo lingo: If you're not familiar, you're in for a treat!
Go to any search engine and type in doggolingo and see all the good bois doing a mlem and a boop.