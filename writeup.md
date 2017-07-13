#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track once without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./images/EpochMse.PNG "Model Mean Squared Error Loss"
[image2]: ./images/left_194.jpg "Example image - left camera"
[image3]: ./images/center_194.jpg "Example image - center camera"
[image4]: ./images/right_194.jpg "Example image - right camera"
[image5]: ./images/recovery_1.jpg "Recovery Example Step 1"
[image6]: ./images/recovery_2.jpg "Recovery Example Step 4"
[image7]: ./images/recovery_3.jpg "Recovery Example Step 8"
[image8]: ./images/center_194.jpg "Example image - center camera"
[image9]: ./images/center_194_flipped.jpg "Example image flipped - center camera"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* finetune.py for fine tuning a trained network
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
My model consists of a convolution neural network modeled after the nvidia car network (model.py lines 33-45).  The first 3 layers are convolution layers with 5x5 filter sizes and depths of 24, 36 & 48 followed by 2 more convolution layers with 3x3 filter sizes and depths of 64.  This is followed by a flatten layer and then 5 fully connected layers with depths 1164, 100, 50, 10, 1.

RELU activations are used for the convolutions to introduce nonlinearity (code lines 34-38).

The data is normalized in the model using a Keras lambda layer (code line 128).

####2. Attempts to reduce overfitting in the model

The model does not contain dropout layers since there was no evidence of overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 133-136). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 133).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Three front facing cameras were used with one mounted center and the others mounted on left and right sides.  I used a combination of center lane driving and recovery driving from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a known model that was likely to work well.  I briefly tried the LeNet architecture but the car didn't drive well.  Perhaps it could have been improved by working more with the dataset but, instead, I opted to switch to the nvidia car network and focus on that.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

With the Nvidia model I first trained the network with 10 epochs and found that mean squared error was continuously decreasing for both the training set and the validation set so I saw little risk of overfitting the model.

![Mean Squared Error][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.  The road course had several different boarders/shoulders including common lane lines, curve markers, dirt and bridge walls.  The car was having problems with the borders that were less frequently encountered.  I surmised that the common lane line markings and curve markings were dominating the data so, to provide balance, extra data was collected for the other borders.  Also, the background for most of the track was hills and trees but one troublesome portion of the track had a blue sky/water background.  Additional data from this portion of the track was collected. 

Numerous training attempts werd made with additional data. Perhaps most importantly, tuning of the steering angle augmentation for the left and right camera images was required.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is modeled after the nvidia car network (model.py lines 33-45).  The first 3 layers are convolution layers with 5x5 filter sizes and depths of 24, 36 & 48 respectively followed by 2 convolution layers with 3x3 filter sizes and depths of 64.  This is followed by a flatten layer and then 5 fully connected layers with depths 1164, 100, 50, 10, 1 respectively.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example images of center lane driving from the 3 front facing cameras:

![Left camera][image2]
![Center camera][image3]
![Right camera][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges of the track should it ever find itself in these situations. These images show what a recovery looks like from the center camera starting from the right side:

![Recovering from right side][image5]
![Recovery step 4][image6]
![Recovery step 8][image7]

To augment the data set, I tried flipping images and angles thinking that this would improve the models. For example, here is an image that has then been flipped:

![Example data][image8]
![Example data flipped][image9]

The flipping, however, didn't appear to improve the training outcome so it was removed.

After the collection process, I had 24541 number of data points. In my keras model I preprocessed this data by first normalizing it to range [-0.5,0.5] and then cropping the data images to exclude the top 40% (sky) and bottom 12% (hood) from the images. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact that mean squared error on the validation set continued to decrease through 10 epochs but training accuracy was no longer improving significantly after the 3rd epoch.

I used an adam optimizer so that manually training the learning rate wasn't necessary.