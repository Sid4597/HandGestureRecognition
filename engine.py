import cv2
import numpy as np
import os
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class engine():
    #Mediapipe Variables
    global mp_holistic, mp_drawing, mp_drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    global actions, no_sequence,sequence_length, DATA_PATH
    DATA_PATH = os.path.join('MP_Data') #path for exported data, numpy arrays
    actions = np.array(['forward', 'backward', 'stop','left','right'])     # actions list
    no_sequence = 30     #30 videos of data
    sequence_length = 30     # 30 frames each

    #Detecting Hands, Pose and Face
    def mediapipe_Detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False 
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        return image, results

    #Draw Landmarks on the body
    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2 ))


    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lh,rh])

    def create_folders():
        for action in actions:
            for sequence in range(no_sequence):
                try:
                    os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
                except:
                    pass


    def load_model():
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True,activation='relu', input_shape=(30,258)))
        model.add(LSTM(128, return_sequences=True,activation='relu'))
        model.add(LSTM(64, return_sequences=False,activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(actions.shape[0],activation='softmax'))

        return tb_callback, model

    def train_mode():
        label_map = {label:num for num, label in enumerate(actions)}

        sequences, labels = [],[]
        for action in actions:
            for sequence in range(no_sequence):
                window=[]
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])
                
        x= np.array(sequences) 
        y = to_categorical(labels).astype(int)
        tb_callback, model = engine.load_model()
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)
        res = [.7,0.2,0.1]
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])
        model.save('action.h5')
        model.summary()

        return x_test, y_test
               
    def model_tests(model, x_test, y_test):
        yhat = model.predict(x_test)
        ytrue = np.argmax(y_test,axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        conf_matrix = multilabel_confusion_matrix(ytrue,yhat)
        accuracy = accuracy_score(ytrue, yhat)
        return conf_matrix, accuracy