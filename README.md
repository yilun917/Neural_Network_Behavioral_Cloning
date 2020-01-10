# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Training_History.jpg "Model Visualization"
[image2]: ./gray_center_2016_12_01_13_42_07_892.jpg "Grayscaling"
[image3]: ./center_2016_12_01_13_42_07_892.jpg "Normal Image"
[image4]: ./flipped_center_2016_12_01_13_42_07_892.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the architecture published by NVIDIA. It mainly consists of 5 convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 68-85) 

The model includes RELU layers to introduce nonlinearity (model.py line 75), and the data is normalized in the model using a Keras lambda layer (model.py line 70). An added grayscale layer will greatly reduce the loss(model.py line 73). Also, a Cropping2D layer was used to reduce the noise of the picture to get a better training result. Top 50 pixes(sky) and bottom 20 pixes (vehicle hood)(model.py line 74). 

#### 2. Attempts to reduce overfitting in the model

Having more than 3 epoches seems to produce the problem of overfitting. However, with just 3 epoches, the model is alreay performing well.
For the purpose of generalizing the model, I still added a dropout layer just before the output layer(model.py line 84).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data
 
 I used the provided training data. All the center, left and right images were used to increase the model accuracy.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to to use the well developed model published by NVIDIA. 

My first step was to process the input data, both normalizng, coverting to gray scale and cropping the unnecessary part. I thought this model might be appropriate because it is proven already on a actual car as introduced by the course.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. While, even with slight overfitting, the vehicle still performs fairly well on the track.

For the purpose of generalizing the model, I modified the model so that it had a dropout layer at the end just before the output layer.

The final step was to run the simulator to see how well the car was driving around track one. Only one spot, the alternate route just after the bridge which doesn't have the right boarder, the vehicle will take the alternate route and hit the tire in the middle. To improve the driving behavior in these cases, I introduced the model training data of left and right camera image. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-88) consisted of a convolution neural network with the following layers and layer sizes: 
Convolution layer of size (24,5,5) with 'relu' activation.
Convolution layer of size (36,5,5) with 'relu' activation.
Convolution layer of size (48,5,5) with 'relu' activation.
Convolution layer of size (64,5,5) with 'relu' activation.
Convolution layer of size (64,5,5) with 'relu' activation.
A flatten layer.
A fully connected layer with 100*1. 
A fully connected layer with 50*1.  
A fully connected layer with 10*1.
A dropout layer with 50% keep rate.
Finally a output layer with 1 output.  


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

Using the provided data, the vehicle doesn't seem to lean much especially with the added flipped image. For example, here is an image that has then been flipped:

![alt text][image4]


After the collection process, I had about 48000 number of data points. I then preprocessed this data by using lambda and cropping layer directly in the model. Here is an example of gray scale image after the coversiont to gray scale lambda layer.

![alt text][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation loss start to increase after that even with drop out layer. I used an adam optimizer so that manually training the learning rate wasn't necessary.

**LICENSE

[MIT LICENSE](./LICENSE)
