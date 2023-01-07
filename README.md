# Smart Accident Reporting
![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/htdocs/logo.png)

# Overview 
The objective of this project is to utilize deep learning and computer vision to identify accidents captured on dashcam footage and notify emergency services by providing them with relevant images of the incident.

# Challenges 
1. The primary challenge we faced was obtaining and categorizing images and videos of accidents into separate groups of 'accident' and 'non-accident' frames.

2. To create a deep convolutional neural network model for this project..

3. Limited computational resorces like GPUs.

# Model Overview
For this project I have tweaked Densenet-161 architecture

Dense Convolutional Network (DenseNet) is a type of feed-forward network that connects each layer to every other layer. Unlike traditional convolutional networks, which have a single connection between each layer and the subsequent layer, DenseNet has L(L+1)/2 direct connections between layers, where L is the number of layers. The feature maps of all preceding layers are used as inputs to each layer, and the feature maps of each layer are used as inputs to all subsequent layers. DenseNets have several benefits, such as mitigating the vanishing gradient problem, enhancing feature propagation, promoting feature reuse, and significantly reducing the number of parameters.

The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.

Model structure    Top-1 error    Top-5 error

densenet121  :  25.35   : 7.83

densenet169  :  24.00   : 7.00

densenet201  :  22.80   : 6.43

densenet161  :  22.35   : 6.20

![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/densenet1.png)


# Prerequisite 

Download anaconda from here https://www.anaconda.com/distribution/#download-section

-   PyTorch

        conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


-   OpenCV

        conda install -c conda-forge opencv
 

-   Dataset of images
    
        https://drive.google.com/file/d/1ydCV5AcdQy6MCWvF3rKRMzzduDuj1RT8/view?usp=share_link


-   Pretrained model binary file

        https://drive.google.com/file/d/1XvaXJFpOp7I0UJ8LEpEPUQ4C3ptXmuhI/view?usp=sharing


# DEMO

***1. Accident***

![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/5.gif)

***2. Non-accident***

![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/6.gif)


# Train 

-   Go to terminal and type

        python train.py

# Tensorboard visual 
-   Traning set 
![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/4.png)

-   Validation set**
![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/2.png)

-   Number of corercts v/s epochs
![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/3.png)


# Test/Accuracy
Go to terminal and type

        python test.py

# Test on video

        python evaluate.py

# Test on Webcam
        python livewebcam.py

# Result

![alt text](https://raw.githubusercontent.com/chaudharyt99/smart-accident-detection-reporting/master/assets/1.png)

The model achieved an 86.00% accuracy rate when tested on a random sample of the video sequences from our dataset, which made up 20% of the total. I plan to retrain the model once I have a good GPU and additional data.
