import os
from typing import Dict, List

import cv2
from data import BodyPart
from data import Person
from data import person_from_keypoints_with_scores
import numpy as np


try:
 
  from tflite_runtime.interpreter import interpreter
except ImportError:
  
  import tensorflow as tf
 inter_var_model_fnc = tf.lite.Interpreter



class Movenet(object):
 '''
 Here we are creating a new class named as Movenet for defining some threshold values related to code , 
 specifically the confidence scores while assessing the predictions. The declared variables with the scores are class instances 
 and these are assigned based upon the confidence the model should have while proceeding with the detected key points.
 '''
  _MINIMUM_CRP_kpt_scr = 0.2
  _TSR_EXP_RTO = 1.9
 _BDY_EXP_RTO = 1.2

  def __init__(self, model_name: str) -> None:
   

   '''
   Here we try to split the model name into its root name and extension. 
   So here we are separating the name such that the root name is assigned to underscore variable (root name not needed in this context)and extension is assigned to “ext” variable. 
   Now function return a tuple , wherein the root is assigned to underscore and extension is assigned to “ext”.
   If the ext variable is empty or does not exist, then it is assigned as “.tflite” which is a generally used extension for Tensorflow model files.
   Then we initialize tensorflow based interpreter object by providing it with model name and setting the threads variable with the variable 4 , 
   wherein this variable tells us about the number of threads to be used for inference (with multi core CPU’s , it works better).
   Next we would allocate memory to tensors so that input and output operations can be performed. 
   After that we fetch all the details regarding the index of input tensor from the interpreter’s input details and.
   The input access is basically needed so that it can be accessed during inference. 
   Analogous to the input inference , output tensor is also needed to access the output tensor for inference. Now 
   using the same approach we get the input details in regards to height and width from the input tensor and store them in respective variables. 
   '''
    _, ext = os.path.splitext(model_name)
    if not ext:
      model_name += '.tflite'

    
   inter_var_model_fnc =inter_var_model_fnc(model_path=model_name, num_threads=4)
   inter_var_model_fnc.allocate_tensors()

    self._input_index =inter_var_model_fnc.get_input_details()[0]['index']
    self._output_index =inter_var_model_fnc.get_output_details()[0]['index']

    self._input_height =inter_var_model_fnc.get_input_details()[0]['shape'][1]
    self._input_width =inter_var_model_fnc.get_input_details()[0]['shape'][2]

    self._interpreter =inter_var_model_fnc
    self._region_of_cropping = None

  def init_region_of_cropping(self, height_of_the_image: int,
                       width_of_the_image: int) -> Dict[(str, float)]:
    """
    In this part of code , we basically want to determine whether the image is landscape oriented or portrait oriented. 
    This helps to basically determine the shape of the bounding box around the image. So initially we check if the width is greater 
    than the height , if it is then its in a landscape mode. So here we set the min x coordinate as 0 and width as 1 , 
    indicating a bounding box across the width of the image. Then we calculate minimum y coordinate and height of bounding box. 
    This center’s around the bounding box ( in a vertical fashion) in the image. Then we calculate box_height . 
    Now all this was done , if the image was landscape oriented. If its not, then box height is set to 1.0 and now bounding box is made 
    according to the portrait mode. Now x_min and box_width is calculated to ensure the correct positioning of bounding box. 
    Finally we return a dictionary of the calculated values such as to define the position of bounding box with respect to 
    image ensuring that the ROI is included in the image.
    """
    if width_of_the_image > height_of_the_image:
      minimum_of_x_value = 0.0
      box_width = 1.0
     min_y_val = (height_of_the_image / 2 - width_of_the_image / 2) / height_of_the_image
      height_of_box = width_of_the_image / height_of_the_image
    else:
     min_y_val = 0.0
      height_of_box = 1.0
      minimum_of_x_value = (width_of_the_image / 2 - height_of_the_image / 2) / width_of_the_image
      box_width = height_of_the_image / width_of_the_image

    return {
        'minimum_of_y':min_y_val,
        'minimum_of_x_value': minimum_of_x_value,
        'maximum_of_y_value':min_y_val + height_of_box,
        'maximum_of_x_value': minimum_of_x_value + box_width,
        'height': height_of_box,
        'width': box_width
    }

  def _torso_visible(self, keypoints: np.ndarray) -> bool:
    """
    Here we intend to get the confidence scores regarding the various body parts and specific key point of the body. 
    These confidence scores are basically an indicator of the fact that how accurately the specific body has been
    detected by the algorithm. We have the array named as “key point” which stores confidence scores and coordinates 
    for the detected key points. Then we compare the values with the minimum required threshold , the output gets 
    stored in the form of Boolean values in the required variables. Here we are comparing it with the required 
    threshold so that we can validate that we have a certain criteria and values are above that threshold. Finally,
    we return the result, here we have a or between two shoulders or hips , this is basically done to give an 
    assurance that at least one out of the two is visible and there is an “and” between the two statements to 
    tell that at least one from each pair is visible . Therefore a “True” is returned otherwise a “False” is returned.
    """
    score_left_hip = keypoints[BodyPart.LEFT_HIP.value, 2]
    score_right_hip = keypoints[BodyPart.RIGHT_HIP.value, 2]
    score_shoulder_left = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
    score_shoulder_right = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

    visibility_left_hip = score_left_hip > Movenet._MINIMUM_CRP_kpt_scr
    visibility_right_hip = score_right_hip > Movenet._MINIMUM_CRP_kpt_scr
    visibility_left_shoulder = score_shoulder_left > Movenet._MINIMUM_CRP_kpt_scr
    visibility_right_shoulder= score_shoulder_right > Movenet._MINIMUM_CRP_kpt_scr

    return ((visibility_left_hip or visibility_right_hip) and
            (visibility_left_shoulder or visibility_right_shoulder))

  def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                      keypoints_of_the_target: Dict[(str, float)],
                                      y_based_center: float,
                                      x_based_center: float) -> List[float]:
    """
    Here we have created a list named as torso_joints which has four elements named as left shoulder ,right shoulder,
    left hip and right hip. Through the import of BodyPart module , we here map different body parts to their detected keypoints 
    containing information. Next, we use iterating variables where we have initialized them to zero and then we iterate over 
    the torso_joints list. For each of the joints , we calculate the vertical and the horizontal distance from the center
    joint (which is the center of the image) to the key points which are stored in list called “target_keypoints”. Now the
    max_torso_yrange and max_torso_xrange is updated, max distances (both vertical and horizontal are stored in the variables)
    .Now we again iterate using a loop over the possible keypoints and then we check for the possible keypoints having a score 
    greater than the threshold value , if its greater then we move with the  keypoint and calculate the max vertical and 
    horizontal distances (from all the keypoints).If the score is less than the threshold score , that part of the body is 
    considered as not visible and loop moves to the next keypoint. Finally all the max keypoint values are returned. This
    provides us with all the information which goes about resizing and cropping the image.
    
    """
    torso_joints = [
        BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_HIP,
        BodyPart.RIGHT_HIP
    ]
    torso_of_yrange_maximum = 0.0
    torso_of_xrange_maximum = 0.0
    for joint in torso_joints:
      distance_of_y = abs(y_based_center - keypoints_of_the_target[joint][0])
      distance_of_x = abs(x_based_center - keypoints_of_the_target[joint][1])
      if distance_of_y > torso_of_yrange_maximum:
        torso_of_yrange_maximum = distance_of_y
      if distance_of_x > torso_of_xrange_maximum:
        torso_of_xrange_maximum = distance_of_x

    body_yrange_maximum_var = 0.0
    body_xrange_maximum_var = 0.0
    for idx in range(len(BodyPart)):
      if keypoints_of_the_target[BodyPart(idx).value, 2] < Movenet._MINIMUM_CRP_kpt_scr:
        continue
      distance_of_y = abs(y_based_center - keypoints_of_the_target[joint][0])
      distance_of_x = abs(x_based_center - keypoints_of_the_target[joint][1])
      if distance_of_y >body_yrange_maximum_var:
       body_yrange_maximum_var = distance_of_y

      if distance_of_x > body_xrange_maximum_var:
        body_xrange_maximum_var = distance_of_x

    return [
        torso_of_yrange_maximum, torso_of_xrange_maximum,body_yrange_maximum_var, body_xrange_maximum_var
    ]

  def _determine_region_of_cropping(self, keypoints: np.ndarray, height_of_the_image: int,
                             width_of_the_image: int) -> Dict[(str, float)]:
    """
    First here we are initializing a empty dictionary where dictionary stores coordinates of detected keypoints,
    where these coordinates are associated with specific information in regards to the body part. Then we calculate 
    pixel coordinates which is related to the coordinates of keypoints array and dimensions of 
    image(which are basically image height and image width). Now we are storing the keypoints (as coordinates) 
    in the target_keypoints list. Next we call the torso visible methods such that we can check the visibility of 
    the keypoints. If the return is true then we calculate the average of the keypoints denoting the average of the 
    keypoints for both the x and y coordinate. Then we call the determine_torso_and_body_range to get the max values 
    for the torso and body in terms of both of x and y range. Then we select the max value out of the 4 values of the 
    calculated manipulations.
    
    """
    
    keypoints_of_the_target = {}
    for idx in range(len(BodyPart)):
      keypoints_of_the_target[BodyPart(idx)] = [
          keypoints[idx, 0] * height_of_the_image, keypoints[idx, 1] * width_of_the_image
      ]

    
    if self._torso_visible(keypoints):
      y_based_center = (keypoints_of_the_target[BodyPart.LEFT_HIP][0] +
                  keypoints_of_the_target[BodyPart.RIGHT_HIP][0]) / 2
      x_based_center = (keypoints_of_the_target[BodyPart.LEFT_HIP][1] +
                  keypoints_of_the_target[BodyPart.RIGHT_HIP][1]) / 2

      (torso_of_yrange_maximum, torso_of_xrange_maximum,body_yrange_maximum_var,
       body_xrange_maximum_var) = self._determine_torso_and_body_range(
           keypoints, keypoints_of_the_target, y_based_center, x_based_center)

      length_cropping_half = np.amax([
          torso_of_xrange_maximum * Movenet._TSR_EXP_RTO,
          torso_of_yrange_maximum * Movenet._TSR_EXP_RTO,
         body_yrange_maximum_var * Movenet._BDY_EXP_RTO,
          body_xrange_maximum_var * Movenet._BDY_EXP_RTO
      ])

     
      '''
      First here we are initializing a empty dictionary where dictionary stores coordinates of detected keypoints,
      where these coordinates are associated with specific information in regards to the body part. Then we calculate 
      pixel coordinates which is related to the coordinates of keypoints array and dimensions of image(which are 
      basically image height and image width). Now we are storing the keypoints (as coordinates) in the target_keypoints 
      list. Next we call the torso visible methods such that we can check the visibility of the keypoints. If the return 
      is true then we calculate the average of the keypoints denoting the average of the keypoints for both the x and y coordinate. 
      Then we call the determine_torso_and_body_range to get the max values for the torso and body in terms of both of x and y range.
      Then we select the max value out of the 4 values of the calculated manipulations.
      
      '''
      distances_to_border = np.array(
          [x_based_center, width_of_the_image - x_based_center, y_based_center, height_of_the_image - y_based_center])
      length_cropping_half = np.amin(
          [length_cropping_half, np.amax(distances_to_border)])

      
      if length_cropping_half > max(width_of_the_image, height_of_the_image) / 2:
        return self.init_region_of_cropping(height_of_the_image, width_of_the_image)
      else:
        length_cropping = length_cropping_half * 2
      corner_cropping = [y_based_center - length_cropping_half, x_based_center - length_cropping_half]
      return {
          'minimum_of_y':
              corner_cropping[0] / height_of_the_image,
          'minimum_of_x_value':
              corner_cropping[1] / width_of_the_image,
          'maximum_of_y_value': (corner_cropping[0] + length_cropping) / height_of_the_image,
          'maximum_of_x_value': (corner_cropping[1] + length_cropping) / width_of_the_image,
          'height': (corner_cropping[0] + length_cropping) / height_of_the_image -
                    corner_cropping[0] / height_of_the_image,
          'width': (corner_cropping[1] + length_cropping) / width_of_the_image -
                   corner_cropping[1] / width_of_the_image
      }
    
    else:
      return self.init_region_of_cropping(height_of_the_image, width_of_the_image)


'''
In this snippet , we are first representing variables such as image ( for input image),
crop_region as a dictionary and crop_size as the desired size of the image. Next we get the max and min values 
(  max and min x and y coordinates) of the cropped region. After that we ensure that  the cop coordinates are within 
the boundary of the image (integer values also ensured). If the min values are , for any reason negative, 
then they are set to zero to prevent cropping outside the image boundary. Max values must not be greater 
than 1 and if it is then it is set to max image dimension to prevent cropping outside the image boundary. 
Based on crop region coordinates and keypoints , padding is done to ensure crop region is resized to desired crop size.
Then cropped region is extracted from the image and then padding is done to the cropped image to ensure cropped 
image can be resized to the needed crop size (and also not losing information). Finally the image is resized 
according to the needed dimensions and returned as a numpy array.

'''

  def _crop_and_resize(
      self, image: np.ndarray, region_of_cropping: Dict[(str, float)],
      crop_size: (int, int)) -> np.ndarray:
    """Crops and resize the image to prepare for the model input."""
   min_y_val, minimum_of_x_value, maximum_of_y_value, maximum_of_x_value = [
        region_of_cropping['minimum_of_y'], region_of_cropping['minimum_of_x_value'], region_of_cropping['maximum_of_y_value'],
        region_of_cropping['maximum_of_x_value']
    ]

    top_region_crp = int(0 ifmin_y_val < 0 elsemin_y_val * image.shape[0])
    bottom_region_crp = int(image.shape[0] if maximum_of_y_value >= 1 else maximum_of_y_value * image.shape[0])
    left_region_crp = int(0 if minimum_of_x_value < 0 else minimum_of_x_value * image.shape[1])
    right_region_crp = int(image.shape[1] if maximum_of_x_value >= 1 else maximum_of_x_value * image.shape[1])

    top_region_padding = int(0 -min_y_val * image.shape[0] ifmin_y_val < 0 else 0)
    bottom_region_padding = int((maximum_of_y_value - 1) * image.shape[0] if maximum_of_y_value >= 1 else 0)
    left_region_padding = int(0 - minimum_of_x_value * image.shape[1] if minimum_of_x_value < 0 else 0)
    right_region_padding = int((maximum_of_x_value - 1) * image.shape[1] if maximum_of_x_value >= 1 else 0)

    output_image = image[top_region_crp:bottom_region_crp, left_region_crp:right_region_crp]
    output_image = cv2.copyMakeBorder(output_image, top_region_padding, bottom_region_padding,
                                      left_region_padding, right_region_padding,
                                      cv2.BORDER_CONSTANT)
    output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

    return output_image

  def _run_detector(
      self, image: np.ndarray, region_of_cropping: Dict[(str, float)],
      crop_size: (int, int)) -> np.ndarray:
    """
    Initially we get the cropped and resized image returned from the function.
    Then we change the datatype of the image to unsigned int. Then we set the input  
    tensor with the image and add a dimension to the image to represent the batch size 
    ( tensorflow needs data in a batch format).Then we perform inference on the training data. 
    Then we get the output tensor with the predicted scores. After squeezing the array we 
    in one dimension, we iterate a loop so that we can update the coordinates of the keypoints
    with scores (based on crop region) and then adjust the y-coordinate ( by scaling) and adding 
    the min y coordinate . Similarly we do this for the x coordinate for the width with regards to 
    image and crop region. Then the updated scores are returned
    
    """

    input_image = self._crop_and_resize(image, region_of_cropping, crop_size=crop_size)
    input_image = input_image.astype(dtype=np.uint8)

    self._interpreter.set_tensor(self._input_index,
                                 np.expand_dims(input_image, axis=0))
    self._interpreter.invoke()

    keypoints_with_scores = self._interpreter.get_tensor(self._output_index)
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

  
    for idx in range(len(BodyPart)):
      keypoints_with_scores[idx, 0] = region_of_cropping[
          'minimum_of_y'] + region_of_cropping['height'] * keypoints_with_scores[idx, 0]
      keypoints_with_scores[idx, 1] = region_of_cropping[
          'minimum_of_x_value'] + region_of_cropping['width'] * keypoints_with_scores[idx, 1]

    return keypoints_with_scores

  def detect(self,
             input_image: np.ndarray,
             reset_region_of_cropping: bool = False) -> Person:
    """
    Here we retrieve the height and width dimensions of the image (this is done using the shape attribute).
    Through this we get 3 outputs , image height , width and ‘_’, meaning the channels in the image (for RGB its 3) 
    and if any crop region is not initialized or its True , then new crop region is in place. 
    The crop region gets initialized ( for the first time) or gets reset based on the input image dimensions.
    Then the pose keypoints (along with the associated scores) are provided for the input image (based on the pose).
    Then we calculate the crop region for the next frame ( it is based on the detected keypoints specifically the scores ,
    image height and width). Lastly , the keypoints with scores , image height and width are returned.
    """
    height_of_the_image, width_of_the_image, _ = input_image.shape
    if (self._region_of_cropping is None) or reset_region_of_cropping:
     
      self._region_of_cropping = self.init_region_of_cropping(height_of_the_image, width_of_the_image)

    
    keypoint_with_scores = self._run_detector(
        input_image,
        self._region_of_cropping,
        crop_size=(self._input_height, self._input_width))
   
    self._region_of_cropping = self._determine_region_of_cropping(keypoint_with_scores,
                                                    height_of_the_image, width_of_the_image)


    return person_from_keypoints_with_scores(keypoint_with_scores, height_of_the_image,
                                             width_of_the_image)