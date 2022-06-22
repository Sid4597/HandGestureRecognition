import cv2
import numpy as np

# import time
# import mediapipe as mp
from engine import engine


class actionSave():

    def create_folders():
        for action in engine.actions:
            for sequence in range(engine.no_sequence):
                try:
                    os.makedirs(os.path.join(engine.DATA_PATH,action,str(sequence)))
                except:
                    pass

    def main():
        actionSave.create_folders()
        cap = cv2.VideoCapture(0)
        with engine.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            #loop through actions
            for action in engine.actions:
                #loop through sequences aka videos
                for sequence in range(engine.no_sequence):
                    #loop through frames
                    for frame_num in range(engine.sequence_length):
                        #Read Feed
                        ret, frame = cap.read()

                        image, results = engine.mediapipe_Detection(frame, holistic)
                        engine.draw_landmarks(image,results)

                        #apply wait logic
                        if frame_num == 0:
                            cv2.putText(image,'Collecting Frames for {} video number {}'.format(action,sequence),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                            (0,0,255),1,cv2.LINE_AA)
                            cv2.putText(image,'STARTING_COLLECTION',(120,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
                            cv2.imshow('Webcam Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image,'Collecting Frames for {} video number {}'.format(action,sequence),(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,
                            (0,0,255),1,cv2.LINE_AA)
                            cv2.imshow('Webcam Feed', image)

                        #Export Keypoints
                        keypoints = engine.extract_keypoints(results)
                        np_path = os.path.join(engine.DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(np_path,keypoints)

                        #end loop
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()

        def create_model(results):
            engine.extract_keypoints(results)
            engine.create_folders()
            x_test, y_test = engine.train_mode()
            conf_matrix, accuracy = engine.create_folders(x_test, y_test)
            print(conf_matrix)
            print(accuracy)