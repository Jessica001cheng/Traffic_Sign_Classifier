# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Visulization.png "Visualization"
[image2]: ./examples/original.png "Original"
[image3]: ./examples/reprocess.png "Grayscaling"
[image4]: ./examples/random_noise.jpg "Random Noise"
[image5]: ./examples/label07.jpg "Traffic Sign 1"
[image6]: ./examples/label14.jpg "Traffic Sign 2"
[image7]: ./examples/label01.jpg "Traffic Sign 3"
[image8]: ./examples/label34.jpg "Traffic Sign 4"
[image9]: ./examples/label00.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Jessica001cheng/Traffic_Sign_Classifier.git/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the nmpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing how the data spread in train/validation/test dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because 
1. To classify traffic signs, shape are more important than colors. So grayscale image is enough for shape detection.
2. Using grayscale image can reduce calculation amount and improve train speed. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image2]


As a last step, I normalized the image data because with normalization, the output of the image is in range[-1.0,1.0]. The calulation will be easier.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Dropout              | keepprobe = 0.8               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x6                |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x6                |
| Dropout              | keepprobe = 0.8               |
| Fully connected		| 120       									|
| RELU                    |    
| Dropout              | keepprobe = 0.8               |
| Fully connected        | 84                                           |
| RELU                    |    
| Dropout              | keepprobe = 0.8               |
| Fully connected        | 43                                           |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet-5 model:
Optimizer: Adam Optimizer
batch size: 128
number of epochs: 10
learning rate:0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.984
* validation set accuracy of 0.944 
* test set accuracy of 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I use the original LeNet-5 model without dropout layer. It is known to me with the last lesson
* What were some problems with the initial architecture?
But after I use, I find the train accuracy is high but validation accuracy is low. I think it is overfitted.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I add dropout layer after conv2D and full connection layer. Then change the keepout parameter. I find the accuracy trend of test and validation are similar.
Also I find after the last full connection layer, if I use RELU activation, the accuracy of test and validation are both low. So I ask in Forum. Below is the link of the question:
https://knowledge.udacity.com/questions/3839
After I remove the RELU activation layer in the last full connection, the accuracy become high.
* Which parameters were tuned? How were they adjusted and why?
I change the dropout parameter from 0.4 to 0.8. I see when it is 0.8, the accuracy is highest.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
1. convolution layer will extract the information(shape, line, etc) from images. So it can work well with this classifier problem.
2. a dropout layer is to help to dropout some data randomly from the batch. So it will resolve the overfit problem. 

If a well known architecture was chosen:
* What architecture was chosen?
LeNet-5 architecture.
* Why did you believe it would be relevant to the traffic sign application?
Because It did well on MNIST data set.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The loss of trained set decrease in every epoch. And accuracy increase in every epoch.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5 ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The 5th image might be difficult to classify because it is dark and have other  shape on it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)     		| Speed limit (30km/h)									| 
| Stop     			| Stop 										|
| Speed limit (30km/h)					| Keep right											|
| Turn left ahead      		| No entry					 				|
| Speed limit (20km/h)			| Speed limit (20km/h)    							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h), and the image does contain a Speed limit (100km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 2.42         			| speed limit (30km/h)   									| 
| 1.04    				| Roundabout mandatory 										|
| 0.71					| End of speed limit (80km/h)											|
| 0.68     			| Speed limit (80km/h)					 				|
| 0.18				    | End of no passing by vehicles over 3.5 metric tons      							|


For the second image Stop sign.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.42         			| Stop   									| 
| 4.37    				| Keep right 										|
| 2.79					| Speed limit (50km/h)											|
| 1.97     			| Speed limit (30km/h)					 				|
| -0.52				    | Speed limit (80km/h)      							|

For the third image is Speed limit (30km/h).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.27         			| Keep right   									| 
| 2.55    				| Roundabout mandatory 										|
| 1.52					| General caution										|
| 1.50     			| Speed limit (30km/h)					 				|
| 1.18				    | Road work      							|

For the 4th image is Turn left ahead.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.64         			| No entry   									| 
| 4.35    				| Stop 										|
| 3.17					| Roundabout mandatory											|
| 2.07     			| Go straight or left					 				|
| 2.01				    | Turn left ahead      							|

For the 5th image is Speed limit (20km/h).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.96         			| Speed limit (20km/h)   									| 
| 8.60    				| Speed limit (30km/h) 										|
| 5.53					| Speed limit (120km/h)											|
| 4.72     			| SSpeed limit (70km/h)					 				|
| 2.75				    | Speed limit (80km/h)      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


