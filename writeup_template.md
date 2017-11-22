#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/traffic_signs_exploration.pbg "Visualization"
[image2]: ./examples/forward_gray.png "Gray Ahead Only Sign"
[image3]: ./examples/forward.png "Ahead Only Sign"
[image4]: ./traffic_signs/30.png "Traffic Sign 1"
[image5]: ./traffic_signs/60.png "Traffic Sign 2"
[image6]: ./traffic_signs/left.png "Traffic Sign 3"
[image7]: ./traffic_signs/right_turn.jpeg "Traffic Sign 4"
[image8]: ./traffic_signs/stop.jpeg "Traffic Sign 5"
[image9]: ./examples/training_data.png "Training & Validation Accuracy"

---
###Writeup / README

Here is a link to my [project code](https://github.com/dooyum/Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
![alt text][image3]
* The number of unique classes/labels in the data set is 43

####Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of examples in the training set belong to each class/label. It is interesting to note that the data is heavily skewed towards speed limit and stop signs.
I considered normalizing the data for equal representation of the signs but the numbers represent real world occurences.
Since this is the case, it will not be bad for the model to predict commonly occuring signs with a higher probability.

![alt text][image1]

###Design and Test of Model Architecture

As a first step, I decided to convert the images to grayscale because the color of the signs are irrelevant to the information they convey i.e. a green stop sign is not different from a red one. This gives the model much less information to take into consideration while classifying the data.

Here is an example of an "ahead only" traffic sign image before and after grayscaling.

![alt text][image3] ![alt text][image2]

As a last step, I normalized the image data because 256 degrees of gray were unnecessary to convey information about the sign.
I therefore normalized the image data with a Min-Max scaling from a range of [0, 255] to a range of [0.1, 0.9].

As an improvement to the model, I plan on generating additional data by creating new images and modifying the brightness and saturation of these images. This way, if all the sample images of a particular sign were taken in the daylight, the model would be able to recognize them in the dark.

All the sample labels were tuned into one-hot tensors over the 43 classes.

####Final model architecture

My final model architecture was a modified LeNet Architecture. I chose to use LeNet as a basis because it performs well on the MNIST dataset which comprises of text based grayscaled images, which are similar to the normalized data. One of the main modifications I made to LeNet was to introduce two dropouts to prevent overfitting.

My final model consisted of the following layers:

| Layer 				| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x24 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x48 	|
| Flatten       	    | outputs 1200 									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				| Droupout rate 0.75							|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Dropout				| Droupout rate 0.75							|
| Fully connected		| outputs 43  									|
| Softmax				| outputs 1x43 matrix of predicted probabilities|
|						|												|
|						|												|
 


####Model Training

To train the model, I used an Adam Optimizer to minimize the training loss.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.3%
* test set accuracy of 94.2%

![alt text][image9]

I computed the training and validation accuracy by evaluating the data sets while the model was training. The main difference was that I used a dropout rate of 100% to ensure all the data was taken into account when evaluating the model.
The accuracy of the test data set was evaluated after the model was saved, in order to ensure my model would perform well in the real world.

My Model's design architecture was based on LeNet. I chose to use LeNet as a basis because it performs well on the MNIST dataset which comprises of text based grayscaled images, which are similar to the normalized and grayscaled traffic sign images.
Pure LeNet gave me a validation accuracy of about 87%. For both stages of pooling I used Max-pooling. For my activation stages I used RELU. A major change I made was to include two dropout functions immediately after each of the last two activation stages. This helped prevent overfitting as I tweeked the model parameters. I chose a dropout rate of 75% in order to ensure the trained model was more robust. After including the dropout stages, the validation accuracy went up to 96.3 while the training accuracy went up to 99.9%.

My initial parameters were: learning rate - 0.0001, batch-size - 50, epochs - 10
I continuosly increased the batch size until my model trained faster but plateaud with regards to accuracy. I made 10x increases to my learning rate, finding the sweet spot between sporadic gradient descents and fast model training. 10 epochs seemed to be too short to train because the model still showed improvements before the final epoch. As a result I increased the number of epochs until the training accuracy hit a plateau.
The final values for my parameters were: learning rate - 0.001, batch-size - 128, epochs - 25

This was a good problem to address with a Convolutional Nueral Network because CNNs work well with image recognition. Each stage of a CNN extracts useful information (e.g. shapes that are only part by a particular sign) to pass on to the next stage. CNNs learn by discarding information that's not really vital to classifying the data e.g. The subtle shade/color of a sign matters much less that the areas of sharp contrast on the sign. A dropout is important in this case because in some epochs, unimportant aspects of the sign are left out of the data and it shows it has little bearing on the signs class.

###Testing my Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because "3" looks like "5" and it might be mistaken for an 80kmph sign.
The second image might be difficult to classify because "6" looks like "8" and it might be mistaken for an 80kmph sign.
The third image might be difficult to classify because the blue color is similar to the blue of the sky behind it.
The fourth image should not be difficult to classify.
The fifth image should not be difficult to classify.

####Prediction Accuracy

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30 km/h)	| Speed limit (30 km/h)							| 
| Speed limit (60 km/h)	| Speed limit (60 km/h)							| 
| Left turn ahead		| Left turn ahead								|
| Stop					| Stop											|
| Right turn ahead		| Right turn ahead								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.2% and the training set of 99.9%

####Prediction Probability

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30 km/h) sign (probability of 0.8), and the image does contain a Speed limit (30 km/h) sign. The second highest probability is a Speed limit (50 km/h) sign, likely because the "3" is similar to a "5". The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| Speed limit (30 km/h)							| 
| .19     				| Speed limit (50 km/h)							|
| .01					| Speed limit (80 km/h)							|
| .00	      			| Wild animals crossing			 				|
| .00				    | Speed limit (60 km/h)							|

For the second image, the model is completely sure that this is a Speed limit (60 km/h) sign (probability of 1.0), and the image does contain a Speed limit (60 km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (60 km/h)							| 
| .00     				| Speed limit (80 km/h)							|
| .00					| Speed limit (30 km/h)							|
| .00	      			| End of speed limit (80km/h)	 				|
| .00				    | Speed limit (50 km/h)							|

For the third image, the model is completely sure that this is a Turn left ahead sign (probability of 1.0), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn left ahead   							| 
| .00     				| Keep right        							|
| .00					| Speed limit (60 km/h)							|
| .00	      			| Children crossing 			 				|
| .00				    | End of all speed and passing limits			|

For the fourth image, the model is completely sure that this is a Stop sign (probability of 1.0), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop              							| 
| .00     				| Yield             							|
| .00					| Keep Right        							|
| .00	      			| Speed limit (50 km/h)			 				|
| .00				    | Speed limit (80 km/h)							|

For the fifth image, the model is almost completely sure that this is a Turn right ahead sign (probability of 0.99), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Turn right ahead  							| 
| .01					| Speed limit (30 km/h)							|
| .00	      			| Roundabout mandatory			 				|
| .00				    | Speed limit (100 km/h)						|
| .00				    | Right-of-way at the next intersection			|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


