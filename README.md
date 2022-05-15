# Sign-Language-Detector-using-Pytorch
Jessica Conroy <br>
UMBC Data Science <br>
Data 690: Practical AI <br>
Final Project <br>

## Contents
Due to the size, the model file and data cannot be uploaded to this repo. All notebooks for generating and annotating data, training the model and saving the model are available in the notebooks folder. Note that you will need to set up labelImgs in order to perform the annotation.

## Overview

The Deaf community and other hearing-impaired individuals often need to resort to carrying notepads and whiteboards with them or to using a translator to communicate with the hearing portions of society. As such, the burden to be understood is placed entirely on the deaf individual and can be especially limited when translators are not available. A program or app that could translate sign language into audible speech could go a long way in improving communication with the hearing population and reduce time and effort spent writing and reading messages. This could be particularly useful in a situation where there are no writing utensils available, translators are unavailable, or when an emergency is in progress and information needs to be shared quickly, such as in the case of an accident or medical emergency. First responders and medical personnel could potentially come prepared to communicate with deaf individuals or patients if this technology exists, and in turn, the Deaf community would be less dependent on translators. 

For this project I attempted to build a sign language detector and translator that uses a video feed to translate simple common sign phrases. The goal was layered, starting with detecting the person in the video, then training a model to identify the words or phrases conveyed by the persons hands, and finally having a text to voice module read the translation back to the user. While the ultimate goal will be to train on a comprehensive dataset, I will start with a few greetings and common phrases and letters.

## Methods and Scope

Data Collection:
Captured images using OpenCV in Python
Created XML Annotations using LabelImg

Image Classification:
Used Pytorch to train an image classification model on Resnet50 (tested Resnet 18-101)
Words/Letters Trained: [‘thank you', 'hello’, 'my', 'name', 'project’, 'this', 'j', 'e', 's’] 
This is a reduced vocabulary than originally planned. Training on the complete alphabet led to very poor performance, in part due to the limited dataset. 
Performance: Test loss: 0.032765, Test accuracy: 1.0000

Realtime Detection:
Used OpenCV to identify hand location in real time, crop the image and feed it into the trained model to produce a prediction. 
![image](https://user-images.githubusercontent.com/63023492/168479455-d6827bb3-045d-486e-b357-3162e2034850.png)

## Results

The resulting application was fairly capable of performing the task at hand. Issues include the difficulty in predicting words requiring two hands and the limited dataset which led to difficulty predicting words when hand orientation. background, clothing, or person changes. 

![image](https://user-images.githubusercontent.com/63023492/168479682-38d6a64d-a279-4b21-bd8d-524bcbc8b37c.png)

## Limitations

#### Limited Data Set

In order to scale this application, the model needs to be run on a much larger dataset with far more letters and words than are represented here. Furthermore the dataset requires more diversity for it to be transferrable to other settings, visual orientations, and clothing, as well as differing gender, racial, and ethnic signers.
At this time, the app will only work for me and others who look like me, and works less well when I change my clothes/backgroud

#### Alternate Methods

While I tried several methods for training this model, there are many others that were not attempted or successful, that, given more time, may have performed better. 
Further optimization and experimentation are warranted to identify the best model for this task, as well as for scaling to larger datasets and use cases.

## Next Steps

#### Train for longer on Larger datasets
Expanded Vocabulary
Expand representation in the dataset (diversity, backgrounds, clothing, etc)

#### Improve the  text to speech element
The current text to speech element says all outputs regardless of correctness. This is confusing. It would therefore help if there was some kind of qualifier or trigger to the speech, such as an amount of time on a particular sign.

#### Build a Translator for the Syntax differences
Sign language does not translate one to one with the English language. Adding a model for translation of complete sentences would go a long way to make this type of app more useable

## Sources: 

https://github.com/nicknochnack/RealTimeObjectDetection
https://github.com/tzutalin/labelImg 
https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/
https://medium.com/academy-eldoradocps/creating-a-custom-neural-network-with-pytorch-fd3621705d32
https://github.com/Gunnika/Sign_Language_Detector-PyTorch
https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
https://towardsdatascience.com/custom-dataset-in-pytorch-part-1-images-2df3152895
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
https://pytorch.org/vision/stable/models.html
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
https://pytorch.org/tutorials/beginner/saving_loading_models.html
https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/faster_rcnn.html 
https://github.com/EdwardRaff/Inside-Deep-Learning 
Raff, E. (2022). Inside Deep Learning: Math, Algorithms, Models (Annotated ed.). Manning. 
https://discuss.pytorch.org/t/collate-issue-with-fast-rcnn-tries-to-transform-dictionary-in-generalizedrcnn/62249 
https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/faster_rcnn.html 
https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html 
https://learnopencv.com/cropping-an-image-using-opencv/ 
https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/ 
https://www.codegrepper.com/code-examples/python/python+to+read+text+aloud\ 
https://python.tutorialink.com/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-detection-just-like-cv2-boundingrect-does/ 
![image](https://user-images.githubusercontent.com/63023492/168479721-63d610d8-1daf-41d2-90c4-fd96e0e1a692.png)


