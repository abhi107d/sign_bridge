import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.keras.models import Sequential
import mediapipe as mp
from tensorflow.keras.layers import LSTM, Dense
import os





def draw(frame,landmarks,mp_draw,mp_hol):
        mp_draw.draw_landmarks(frame,landmarks.face_landmarks,mp_hol.FACEMESH_CONTOURS,
                              mp_draw.DrawingSpec(color=(98,226,34), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(238,38,211), thickness=1, circle_radius=1))
        mp_draw.draw_landmarks(frame,landmarks.left_hand_landmarks,mp_hol.HAND_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(238,231,38), thickness=3, circle_radius=4),
                              mp_draw.DrawingSpec(color=(238,38,211), thickness=3, circle_radius=4))
        mp_draw.draw_landmarks(frame,landmarks.right_hand_landmarks,mp_hol.HAND_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(238,231,38), thickness=3, circle_radius=4),
                              mp_draw.DrawingSpec(color=(238,38,211), thickness=3, circle_radius=4))
        mp_draw.draw_landmarks(frame,landmarks.pose_landmarks,mp_hol.POSE_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(98,226,34), thickness=2, circle_radius=2),
                              mp_draw.DrawingSpec(color=(238,38,211), thickness=2, circle_radius=2))
        
def extract_landmarks(landmarks):
    if landmarks.pose_landmarks:
        pose=np.array([[p.x,p.y,p.z,p.visibility] for p in landmarks.pose_landmarks.landmark]).flatten()
    else:
        pose=np.zeros(132,)
    if landmarks.left_hand_landmarks:
        left_hand=np.array([[p.x,p.y,p.z] for p in landmarks.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand=np.zeros(63,)
    if landmarks.right_hand_landmarks:
        right_hand=np.array([[p.x,p.y,p.z] for p in landmarks.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand=np.zeros(63,)
    if landmarks.face_landmarks:
        face=np.array([[p.x,p.y,p.z] for p in landmarks.face_landmarks.landmark]).flatten()
    else:
        face=np.zeros(1404,)

    return np.concatenate([face,pose,left_hand,right_hand])


label_map=["idel","hello","thanks"]

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(29,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights(os.path.join("models","main.h5"))        


mp_hol=mp.solutions.holistic
mp_draw=mp.solutions.drawing_utils
#just capture
cam=cv2.VideoCapture(0)
action=[]
text=[]
# predictions = []
trsh=0.6
res=np.array([0,0])

with mp_hol.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    while cam.isOpened():
        ret,frame=cam.read()
        if not ret:
            break
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        landmarks=hol.process(image)

        points=extract_landmarks(landmarks)

        #getting 30 frames of action
        action.append(points)
        action=action[-29:]
        if len(action)==29:
            res=model.predict(np.expand_dims(action,axis=0),verbose=0)[0]
            # predictions.append(np.argmax(res))
            p_idx=np.argmax(res)
            # predictions=predictions[-10:]
            # print(predictions)
            # if np.unique(predictions)[-1]==np.argmax(res): 
               
            if  res[p_idx]>trsh:
                if len(text)>0:
                    if label_map[p_idx]!=text[-1]:
                        text.append(label_map[p_idx])
                        text=text[-1:]
                        
                else:
                    text.append(label_map[p_idx])

                
            
            if len(text)>5:
                text=text[-5:]
                
                
              
           
        #drawing on image
        draw(frame,landmarks,mp_draw,mp_hol)
        frame=cv2.flip(frame,1)

        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(text), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("capture",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()