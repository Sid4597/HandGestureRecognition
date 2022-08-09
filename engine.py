#!/usr/bin/env python3
"""
******************************************
Title       : Gesture Recognition Engine
Description : Hand Gesture Recogniting functions
Date        : 2-9-2022
Author      : Siddharth Shaligram
Version     : 0.0
******************************************
Change Log:
******************************************
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class Engine(object):
    """docstring for Engine."""

    def __init__(self, arg):
        super(Engine, self).__init__()
        self.arg = arg
        #Mediapipe Variable
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.DATA_PATH = os.path.join('MP_Data') #path for exported data, numpy arrays
        self.actions = np.array(['forward', 'backward','left','right','spin','stop','rest'])     # actions list
        self.no_sequence = 30     #30 videos of data
        self.sequence_length = 30     # 30 frames each

    #Detecting Hands, Pose and Face
    def mediapipe_Detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        return image, results

    #Draw Landmarks on the body
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2))
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2 ))


    def extract_keypoints(self,results):
        # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        # return np.concatenate([pose,lh,rh])
        return rh

    def create_folders(self):
        for action in actions:
            for sequence in range(no_sequence):
                try:
                    os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
                except:
                    pass


    def load_model(self,name=""):
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)


        model = Sequential()
        model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(30,63)))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(64, return_sequences=False,activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(self.actions.shape[0],activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        if name != "":
            name = '/home/lunet/cosms2/catkin_ws/src/project/scripts/'+name
            model.load_weights(name)

        return tb_callback, model

    def train_mode(self, model):
        label_map = {label:num for num, label in enumerate(self.actions)}

        sequences, labels = [],[]
        for action in self.actions:
            for sequence in range(self.no_sequence):
                window=[]
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        x= np.array(sequences)
        y = to_categorical(labels).astype(int)
        tb_callback, model = self.load_model()
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)
        res = [.7,0.2,0.1]
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(x_train, y_train, epochs=3000, callbacks=[tb_callback])
        model.save('actionv3.h5')
        print(model.summary())

        return x_test, y_test

    def model_tests(self,model, x_test, y_test):
        yhat = model.predict(x_test)
        ytrue = np.argmax(y_test,axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        conf_matrix = multilabel_confusion_matrix(ytrue,yhat)
        acc = 'xx'
        # loss, acc = model.evaluate(x_test, y_test, verbose=2)
        accuracy = accuracy_score(ytrue, yhat)
        return conf_matrix, accuracy, acc
