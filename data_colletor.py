import numpy as np
import cv2 
import os
import mediapipe as mp


mp_hol=mp.solutions.holistic
mp_draw=mp.solutions.drawing_utils

def draw(frame,landmarks,mp_draw,mp_hol):
        mp_draw.draw_landmarks(frame,landmarks.face_landmarks,mp_hol.FACEMESH_CONTOURS,
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1))
        mp_draw.draw_landmarks(frame,landmarks.left_hand_landmarks,mp_hol.HAND_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1))
        mp_draw.draw_landmarks(frame,landmarks.right_hand_landmarks,mp_hol.HAND_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1))
        mp_draw.draw_landmarks(frame,landmarks.pose_landmarks,mp_hol.POSE_CONNECTIONS,
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                              mp_draw.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1))
                              
                              
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
                                        



#adjust values here
path_=os.path.join("Data")
no_frames=30
action=input("\nenter the action label to record\neg hello/thanks etc: ")
start=int(input("Enter start video number\neg if last video in "+action+" folder is 10 enter 11 here : "))
no_video=int(input("Enter the number of video's to record\n any number : "))


for vid in range(start,no_video+start):
    try:
        os.makedirs(os.path.join(path_,action,str(vid)))
    except:
        pass
    
cam=cv2.VideoCapture(0)
flag=False
with mp_hol.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hol:
    
    for vid in range(start,start+no_video):
        if flag:
            break
        for frame_no in range(no_frames+1):
            
        
            ret,frame=cam.read()
            if not ret:
                break
            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            landmarks=hol.process(image)
                
            #drawing on image
            draw(frame,landmarks,mp_draw,mp_hol)
            frame=cv2.flip(frame,1)
            
            if frame_no==0:
                cv2.putText(frame,"Starting collection",(0,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),4,cv2.LINE_AA)
                cv2.putText(frame,"collecting frame for {} video no {} Frame no {}".format(action,vid,frame_no),(120,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                cv2.imshow("capture",frame)
                if cv2.waitKey(50000) or 0xFF == ord('c'): #change capture technique
                    continue
            else:
                cv2.putText(frame,"collecting frame for {} video no {} Frame no {}".format(action,vid,frame_no),(120,20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

            
            feature=extract_landmarks(landmarks)
            path=os.path.join(path_,action,str(vid),str(frame_no-1))
            np.save(path,feature)

            
                
            cv2.imshow("capture",frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                flag=True
                break
    
    cam.release()
    cv2.destroyAllWindows()






                                

