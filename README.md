# Who-s-that-good-boi
* Exploratory data sciencing to get a decent dog predicter using CNN's and transfer learning from one of the good architechtures on keras
	* Keep model the same, cut off dense layers and add 3 new
* Web app to predict "Who's that good boi?" Pokemon style
	* Either camera accescibility if not hard
	* or just upload pictures/paste URLs
	* Pokemon animation/silluoet?
	* Deploy on heuroku



## Overview
* Using the ResNet50 model to classify if it's a dog or not
* If human post something funny (what dog resemples)
* If dog then guess 

* preload resnet50 with imagenet weights
* use standard to predict humans and dogs
* use transfer learning to predict races
	* small dataset, so keep weights, but cutoff ending layers and retrain those
	
	
	
"Heck ur doing me a bamboozle fren"