# Who-s-that-good-boi
* Exploratory data sciencing to get a decent dog predicter using CNN's and transfer learning from one of the good architechtures on keras
	* Keep model the same, cut off dense layers and add 3 new
* Web app to predict "Who's that good boi?" Pokemon style
	* Either camera accescibility if not hard
	* or just upload pictures/paste URLs
	* Pokemon animation/silluoet?
	* Deploy on heuroku

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
To classify human/no-human 


* Using the ResNet50 model to classify if it's a dog or not
* If human post something funny (what dog resemples)
* If dog then guess 

* preload resnet50 with imagenet weights
* use standard to predict humans and dogs
* use transfer learning to predict races
	* small dataset, so keep weights, but cutoff ending layers and retrain those
	
## Analysis

## Conclusions

	



## Packages and other recourses
* OpenCV 4.20 (pip install opencv-python)
* Flask 1.1.1
* Imagenet labels https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
* Dog dataset https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

## Instructions
* cd to app folder
* serve the app by executin python app.py
* Open a browser and enter your local ip address
* Make sure port 5000 is allowed in your firewall

## Etc.
A note on doggo lingo: If you're not familiar, you're in for a treat!
Go to any search engine and type in doggolingo and see all the good bois doing a mlem and a boop.