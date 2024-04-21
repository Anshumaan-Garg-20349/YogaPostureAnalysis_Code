'''
The above libraries in  are needed for the following tasks:-

Cv2 - Tasks such as image processing , reading images , resizing , filtering and
detecting objects are done using this library

Math - Mathematical functions are used from this library for manipulating the data frames 
and angles so as to get the correct modified_results

Numpy - It is used for numerical computations and  array handling

Pandas - It is used to handle large data frames wherein function are directly applied to whole data 
frames and pandas also handle data manipulation and analysis

Matplotlib.pyplot - This library is used to visualize the modified_results through graphs and visual 
representations thereby helping in taking better decisions

Ultralytics - A deep learning framework and which has given significance to object detection and 
analysis. This also involves tasks related to model training , taking inference and overall evaluation

From Ultralytics import Yolo - This line of code directly imports YOLO model from the ultralytics library.
This indicates that code intends to use the YOLO model for classifying and analyzing the images or video.

'''
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

path_initial = r'PATH TO IMAGE'

img_path_var=plt.imread(path_initial)
plt.imshow(img_path_var)
plt.axis('off')
plt.show()

!yolo pose predict model=yolov8n-pose.pt source={path_initial} 

path_secondary = r'PATH TO IMAGE'

img_path_var=plt.imread(path_secondary)
plt.imshow(img_path_var)
plt.axis('off')
plt.show()

'''
Now here we load the model (pose model of yolo) and apply the modified_results on the image. 
Next, we have initialized a data frame with the help of pandas data frame function 
which gives us 6 columns in the dataset. Then we run a for loop where we iterate it 
for the length of modified_results variable and extract all the information returned by the 
bounding box such as the confidence score , coordinates and class labels. We convert 
this data into float datatype. After that we add a new column in the data frame , 
named as ‘i’ and assign the iterating variable value each time , thereby keeping a track of 
iteration. Next, we intend to concatenate the data from the two data frames into one and along 
row(axis=0).After that we assign the column names to columns , which are the coordinates , 
confidence scores and index of the detection result. Finally, we show the extracted box 
information from the modified_results in a tabular format

'''

model = YOLO('yolov8n-pose.pt')  
modified_results = model(path_initial)  

BOX=pd.DataFrame(columns=range(6))
for i in range(len(modified_results)):
    arri=pd.DataFrame(modified_results[i].boxes.data).astype(float)
    arri['i']=i
    BOX=pd.concat([BOX,arri],axis=0)
BOX.columns=['x','y','x2','y2','confidence','class','i']
display(BOX)

'''
The first line , access the information of bounding box result which is 
returned by YOLO model and “.boxes.data” , this is one attribute which 
contains information about the bounding boxes which were there in the result 
and were detected by model such as confidence scores , coordinates and 
class labels. Now we tend to see the key points which were in the result 
for the specific posture and the third line tends to access the confidence 
score with regards to key points. Next line gives the key points data and 
converts it into a NumPy based array and then .data gives the coordinates. 
Extraction of all the detected key points with all the rows and only initial 
2 columns of the NumPy array gives the x and y coordinates of the key points

'''

modified_results[0].boxes.data


modified_results[0].keypoints

modified_results[0].keypoints.conf

numpy_array = modified_results[0].keypoints.data[0].numpy()
coordinate_var = numpy_array[:, 0:2]
print(coordinate_var)

'''
Here we are separating the x and y coordinates into two separate lists 
specifically for plotting purpose. Again, we read the path to the result 
image and display it to the user and put the axis in an off state. We can 
use the scatter plot here to plot the modified_results and see the trends among 
the coordinates where the parameters are entered as “x1_coordinate” for 
x coordinates , “y1_coordinate” as y coordinates , color as blue and 
marker as o. Then we initiate a loop which iterates across each coordinate pair , 
and a text label across to the plot specified by the x and y coordinate 
( which are basically the coordinates of the keypoints) . The other parameters 
specify the alignments arguments relative to the position which can be displayed 
corresponding to the point. Then we print the plot to be shown to the users and 
identify the points being plotted (against the coordinates)

'''


x1_coordinate, y1_coordinate = zip(*coordinate_var)

img_path_var=plt.imread(path_secondary)
plt.imshow(img_path_var)
plt.axis('off')
plt.scatter(x1_coordinate, y1_coordinate, color='blue', marker='o')


for i, (x, y) in enumerate(coordinate_var):
    plt.text(x, y, str(i+1), ha='right', va='bottom')


plt.show()



'''
There are 3 coordinates which are basically 3 points in a 2D space. 
There is a dot product calculation which basically involves 
the angle (theta) part measuring angles between two vectors. 
Magnitude1 and Magnitude2 are basically the magnitudes of vector1 and vector2. 
Then we calculate the cosine angle between the two vectors by dividing the dot 
product and product of two vector magnitudes. The angle between the vectors 
in cosine form is converted to radians and clip function ensures the range 
of the angle (passed to the function) remains between -1 and +1 due to the 
floating point errors. At the last , radian angle is converted to degrees 
via the np.degrees function
'''

def angle_calc(coord_first, coord_second, coord_third):
    vector_first = np.array(coord_first) - np.array(coord_second)
    vector_second = np.array(coord_third) - np.array(coord_second)

    dot_based_product = np.dot(vector_first, vector_second)
    first_magnitude = np.linalg.norm(vector_first)
    second_magnitude = np.linalg.norm(vector_second)

    angle_of_cosine = dot_based_product / (first_magnitude * second_magnitude)


    angle_of_rad = np.arccos(np.clip(angle_of_cosine, -1.0, 1.0))
    angle_of_deg = np.degrees(angle_of_rad)

    return angle_of_deg

''''
Here we have listed the example coordinates , then we iterate the for 
loop across the coordinates and then call the calculate angle function 
passing the coordinates , continuously iterating for all coordinates 
and printing angles between continuous 3 coordinates in degrees

'''


coord_one = [345.3885, 156.2107]
coord_two = [289.2111, 267.8549]
coord_three = [312.9456, 257.4545]


for i in range(len(coordinate_var) - 2):
    angle = angle_calc(coordinate_var[i], coordinate_var[i+1], coordinate_var[i+2])
    print(f"Angle formed between the corresponding coordinates {i+1},{i+2},{i+3}: {round(angle, 2)} ( in degrees)")
    
'''
Here the information is provided for the “orig_shape” attribute
and wherein the original shape of the image is provided in terms 
of width and height. This function provides the information about 
the image before any kind of resizing and processing is done on the 
image. Next we convert the coordinates of the modified_results (keypoints) 
into a numpy array and we assign it to a variable boxes. Next we 
print the coordinates (keypoints) using the print function with a 
size specified in the functions and then we plot a scatter plot , 
to see the trend and correlation between the parameters. Here we plot 
against the x coordinates and the negative y coordinates 
( negative y coordinates are usually done to invert the y axis , 
which is a common thing done in computer vision so that origin can 
match the top left corner). Then we display the scatter plot to the 
user with the keypoints on the image.
'''

modified_results[0].keypoints.orig_shape
print(modified_results[0].keypoints.xy)
modified_boxes = np.array(modified_results[0].keypoints.xy)
print(modified_boxes.shape)
plt.figure(figsize=(6,3))
plt.scatter(modified_boxes[0][:,0], -modified_boxes[0][:,1], )
plt.show()

print(modified_results[0].keypoints.xyn)
modified_boxes = np.array(modified_results[0].keypoints.xyn)
print(modified_boxes.shape)
plt.figure(figsize=(5,7))
plt.scatter(modified_boxes[0][:,0], -modified_boxes[0][:,1], )
plt.show()

print(modified_results[0].modified_boxes.data)
img_path_var=plt.imread(path_secondary)
plt.imshow(img_path_var[50:329,99:556])
plt.show()