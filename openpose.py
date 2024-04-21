'''
Necessary libraries are imported such as cv2 for image/video processing related tasks. 
Next is numpy which is used for array based manipulations of vectors and argparse will 
be used to handle command line arguments
'''
import cv2 as cv
import numpy as np
import argparse

'''
Initially we are creating a object of argparse to handle command line arguments and 
then we are defining 4 command line arguments such as input, the , width and height.
Now we store the command line arguments in the args object. After that we map different
body parts to there indices and make related pose pairs such that angles can be calculated between them.

'''
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='OpenPose\self_image.jpg')
parser.add_argument('--thr', default=0.09, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=328, type=int, help='Resize input to a specific width.')
parser.add_argument('--height', default=328, type=int, help='Resize input to a specific height.')

args = parser.parse_args()


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17 }

POSE_PAIRS=[ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

'''
Next we set the input image dimensions by taking input width and input height variables 
and equating it the values of width and height in the args object. Next we load the trained 
model and load the input image. If the image cannot be loaded due to some reason then error 
message is displayed. Now we get the input image dimensions and then we prepare the input 
image for the model
'''
inWidth = args.width
inHeight = args.height


net = cv.dnn.readNetFromTensorflow("D:\Major Project\Major Project Algorithms\OpenPose\graph_opt.pb")


image = cv.imread(args.input)

if image is None:
    print("Error: Could not load the input image.")
    exit(1)

frameWidth = image.shape[1]
frameHeight = image.shape[0]


blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(inWidth, inHeight), mean=(255, 255, 255), swapRB=True, crop=False)


'''
Next we set the input for the model and then we check the length of the body 
parts and heatmap output dimensions. Now we extract keypoints from the heatmap and then 
define a function to extract angles from the heatmap

'''

net.setInput(blob)


out = net.forward()
out = out[:, :18, :, :]  

assert(len(BODY_PARTS) == out.shape[1])

points = []
for i in range(len(BODY_PARTS)):
    
    heatMap = out[0, i, :, :]

    
    _, conf, _, point = cv.minMaxLoc(heatMap)

   
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]

    
    points.append((int(x), int(y)) if conf > args.thr else None)
    
'''
In the function to calculate the angles , we take the keypoints into an array and 
convert them into a vector. Now from these vectors , cosine angle is calculated 
using the dot product. This angle is finally converted to radians to get the required angle. 
Then the two keypoints are extracted for each pair and this ensures they are valid body parts. 
After that a list is constructed which consists of first key point (in this case it “Neck”) and 
then the second key point. Then the calculate angle function is called so that angles can be calculated 
depending upon the pose pairs

'''

def calculate_angle(keypoint1, keypoint2, keypoint3, points):
    if points[BODY_PARTS[keypoint1]] and points[BODY_PARTS[keypoint2]] and points[BODY_PARTS[keypoint3]]:
        kp1 = np.array(points[BODY_PARTS[keypoint1]])
        kp2 = np.array(points[BODY_PARTS[keypoint2]])
        kp3 = np.array(points[BODY_PARTS[keypoint3]])

        vector1 = kp1 - kp2
        vector2 = kp3 - kp2

        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg % 360

text_y = 40

for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    angle_keypoints = [partFrom, "Neck", partTo]
    angle = calculate_angle(*angle_keypoints, points)
    
'''
Here we are checking if the angle calculation was successfully done so that , 
we display the required text otherwise corresponding text is displayed. 
The text_y variable is then incremented so that vertical position is adjusted 
and next annotation can be processed. Keypoints are then drawn on the image and 
drawing filled circles at the coordinate position. Then finally the pose pairs based 
keypoints and corresponding key point ID’s are connected.

'''

    if angle is not None:
        angle_text = f"Angle ({partFrom} to {partTo}): {angle:.2f} degrees"
        print(angle_text)
        cv.putText(image, angle_text, (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        text_y += 20  
    else:
        angle_text = f"Angle ({partFrom} to {partTo}): Not detected"
        cv.putText(image, angle_text, (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        text_y += 20 


for i, point in enumerate(points):
    if point:
        x, y = point
        cv.circle(image, (x, y), 4, (0, 0, 255), thickness=-1, lineType=cv.FILLED)
        cv.putText(image, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


cv.imshow('OpenPose using OpenCV', image)
cv.waitKey(0)
cv.destroyAllWindows()