import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.nn import Flatten
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import time

os.getcwd()

idx_to_class = {0: 'thank you',
                1: 'e',
                2: 'hello',
                3: 'j',
                4: 'my',
                5: 'name',
                6: 'project',
                7: 's',
                8: 'this'}

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ObjectClassifier(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectClassifier, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses

        # build the classifier head to predict the class labels
        self.classifier = Sequential(
            Linear(2048, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses)
        )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        classLogits = self.classifier(features)
        # return the outputs as a tuple
        return (classLogits)


# Load in model
resnet = resnet50(pretrained=True)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
    param.requires_grad = False

ObjectClassifier = ObjectClassifier(resnet, len(idx_to_class))
# load saved state
checkpoint_dict = torch.load('classification_200epochs_croppedTI_checkpoint (2).pth', map_location=device)
ObjectClassifier.load_state_dict(checkpoint_dict['model_state_dict'])
ObjectClassifier.eval()

# get transforms used in training
Transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# build and start detector
import pyttsx3, time
import cv2
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        if len(hand_landmarks) == 2:  # When there are two hands in the view, I want a single box containing both hands
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            combined = []
            for handLMs in hand_landmarks:
                for lm in handLMs.landmark:  # create combined list of hand landmarks
                    combined.append(lm)
            for lm in combined:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            # Crop based on identified hands and pas through model for prediction

            try:
                image = frame[x_max + 10:y_max + 10, x_min - 10:y_min - 10]  # crop to just hands
                #             print(image)
                #             try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = Transform(image)
                image = image.unsqueeze(0)
                print(image)
                # determine the class label with the largest predicted
                # probability
                prediction = ObjectClassifier(image)
                #             print(prediction)
                prediction = torch.nn.Softmax(dim=1)(prediction)
                #             print(prediction)
                i = prediction.argmax(dim=-1).cpu()
                label = idx_to_class[i.item()]
            except:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = Transform(image)
                image = image.unsqueeze(0)
                #                 print(image)
                # determine the class label with the largest predicted
                # probability
                prediction = ObjectClassifier(image)
                #             print(prediction)
                prediction = torch.nn.Softmax(dim=1)(prediction)
                #             print(prediction)
                i = prediction.argmax(dim=-1).cpu()
                label = idx_to_class[i.item()]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0), 1)
            engine = pyttsx3.init()
            engine.say(label)
            engine.runAndWait()
        #             mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
        else:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                # Crop based on identified hands and pas through model for prediction
                image = frame[x_max - 10:y_max + 10, x_min - 10:y_min + 10]  # crop to just hands
                print(image)
                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    image = Transform(image)
                    image = image.unsqueeze(0)
                    #     print(image)
                    # determine the class label with the largest predicted
                    # probability
                    prediction = ObjectClassifier(image)
                    #             print(prediction)
                    prediction = torch.nn.Softmax(dim=1)(prediction)
                    #             print(prediction)
                    i = prediction.argmax(dim=-1).cpu()
                    label = idx_to_class[i.item()]
                except:
                    label = 'No Sign'
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0), 1)
                if label != 'No Sign':
                    engine = pyttsx3.init()
                    engine.say(label)
                    engine.runAndWait()
    #                 mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    # show the output image
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
