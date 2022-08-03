import cv2
import numpy as np
from engine import Engine
# from RobotClass import Robot
import mediapipe as mp
from scipy import stats

arg = []
eng = Engine(arg)

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

def main():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    colors = [(245,117,16), (117,245,16), (16,117,245), (255, 255, 102),(51, 204, 255),(0, 51, 102),(255, 153, 0)]
    tb_callback, model = eng.load_model('actionv1.h5')

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with eng.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = eng.mediapipe_Detection(frame, holistic)

            # Draw landmarks
            eng.draw_landmarks(image, results)

            # 2. Prediction logic
            keypoints = eng.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(res)
                predictions.append(np.argmax(res))


            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if Engine.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(Engine.actions[np.argmax(res)])
                        else:
                            sentence.append(Engine.actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, eng.actions, image, colors)

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('Webcam Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
   main()
