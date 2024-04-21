
'''
In the set of above libraries , cv2 was needed image processing which includes loading, displaying ,
manipulating and analyzing images. Then we mediapipe framework which is needed to track human body 
poses through images.numpy is used for array and vector manipulation. The library pyttsc3 is used for 
text to speech conversion of a text in a file
'''
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

'''
Here we are initializing a init function and then we define text2speech function 
where we read a text file and set the speech rate limit at 150. The text is converted to speech 
and finally calls the runAndWait function to ensure that the voice/speech is played in a consistent manner
'''
engine = pyttsx3.init()
def text2speech(text_file):
    posture=text_file
    f=open(posture,'r')
    text=f.read()
    f.close()
    engine.setProperty('rate',150)
    engine.say(text)
    engine.runAndWait()
    
'''
The above code defines the function to calculate angle using the 3 points 
(numpy arrays) and this angle calculation is done using the arctan2 function
which converts the angle from radians to degrees. The angle is ensured to be in the range 0 to 180

'''
def calculate_angle(a,b,c):
    vector_first = np.array(a) 
    vector_second = np.array(b) 
    vector_third = np.array(c)
    
    angle_in_radians = np.arctan2(vector_third[1]-vector_second[1], vector_third[0]-vector_second[0]) - np.arctan2(vector_first[1]-vector_second[1], vector_first[0]-vector_second[0])
    new_angle_modified = np.abs(angle_in_radians*180.0/np.pi)
    
    if new_angle_modified >180.0:
        new_angle_modified = 360-new_angle_modified
        
    return new_angle_modified


'''
Here we capture the user posture using the video function and label for 
any undetected pose would be “Undetected pose”. It read the frames from 
the read function and converts the color format from BGR to RGB . 
The RGB image is then used to detect and track human poses. Now after 
tracking and detection is done , the image format is again converted back to BGR , 
then we try to access the landmarks across the posture

'''
cap = cv2.VideoCapture(0)


label="Unknown Posture"


with mp_pose.Pose(minimum_required_det_confi=0.5, minimum_required_trac_confi=0.5) as posture:
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
       
        modified_result = posture.process(image)
    
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
       
        try:
            modified_landmarks = modified_result.pose_landmarks.landmark
            
            
    '''
    Now we access the landmark from the results, retrieve the coordinates of 
    body joints. We start by calculating angles between the joints> Next we do the pose 
    classification based on the  joint angles , such if the elbows are straight enough , 
    knees are also straight or bent according to the posture. This demarcation is also 
    done based on the angles range. Now as and when the classification is done , we update
    the label variable for the detected pose and call the text to speech to provide the 
    feedback based on audio. If the pose is not from any of the detected pose , label is 
    marked as Undetected pose. Finally detections are rendered and displayed to the user.
    '''
    
            shoulder_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_left_processing = [modified_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,modified_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder_right_processing= [modified_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right_processing = [modified_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_wrist_processing = [modified_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip_right_processing = [modified_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right_processing = [modified_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_right_processing = [modified_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,modified_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            
            angle_left_elbow_processing = calculate_angle(shoulder_left_processing, elbow_left_processing, wrist_left_processing)
            right_elbow_angle=calculate_angle(shoulder_right_processing,elbow_right_processing,wrist_left_processing)
            angle_processing_left_shoulder = calculate_angle(hip_left_processing, shoulder_left_processing, elbow_left_processing)
            right_shoulder_angle_processing = calculate_angle(hip_right_processing, shoulder_right_processing, elbow_right_processing)
            hip_left_processing_angle=calculate_angle(knee_left_processing,hip_left_processing,shoulder_left_processing)
            right_hip_angle=calculate_angle(knee_right_processing,hip_right_processing,shoulder_right_processing)
            angle_left_processing_knee=calculate_angle(ankle_left_processing,knee_left_processing,hip_left_processing)
            angle_of_ryt_knee=calculate_angle(ankle_right_processing,knee_right_processing,hip_right_processing)
           
            
            if angle_left_elbow_processing > 165 and angle_left_elbow_processing < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
                if angle_processing_left_shoulder > 80 and angle_processing_left_shoulder < 110 and right_shoulder_angle_processing > 80 and right_shoulder_angle_processing < 110:
                    
                    if angle_left_processing_knee > 165 and angle_left_processing_knee < 195 or angle_of_ryt_knee > 165 and angle_of_ryt_knee < 195:
                        if angle_left_processing_knee > 90 and angle_left_processing_knee < 120 or angle_of_ryt_knee > 90 and angle_of_ryt_knee < 120:
                            label="Posture of Warrior"
                            text2speech("PATH TO TEXT FILE")
                    
                    if angle_left_processing_knee > 160 and angle_left_processing_knee < 195 and angle_of_ryt_knee > 160 and angle_of_ryt_knee < 195:
                        label = 'POSTURE OF T'    
                        text2speech("PATH TO TEXT FILE")
        
            if angle_left_processing_knee > 165 and angle_left_processing_knee < 195 or angle_of_ryt_knee > 165 and angle_of_ryt_knee < 195 :
                if angle_left_processing_knee > 315 and angle_left_processing_knee < 335 or angle_of_ryt_knee > 25 and angle_of_ryt_knee < 45 :
                    label = 'POSTURE OF TREE'
                    text2speech("PATH TO TEXT FILE")
                    
            else:
                label = 'Unknown posture'

            print("Angle of Right elbow:",right_elbow_angle)
            print("Angle of left elbow:",angle_left_elbow_processing)
            print("Right  ")
        
        except:
            pass
        
        
        cv2.putText(image, label,(10,60),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 1, cv2.LINE_AA)

        
        
        
        mp_drawing.draw_landmarks(image, modified_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,155), thickness=3, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,125,0), thickness=3, circle_radius=4))
                
        cv2.imshow('Result', image)

        if cv2.waitKey(26) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()