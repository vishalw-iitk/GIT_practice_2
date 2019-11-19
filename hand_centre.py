from hand_util.utils import detector_utils as detector_utils
from cv2 import cv2
import tensorflow as tf
import numpy as np

detection_graph, sess = detector_utils.load_inference_graph()
num_hands_detect = 2
score_thresh = 0.35
12345
355
5egrg
dgrv
def detect(image):123
    blank_image = np.zeros((260,210,3), np.uint8)
    im_height,im_width = image.shape[:2]
    boxes, scores = detector_utils.detect_objects(image,detection_graph, sess)
    temp1_mpx = 0
    temp2_mpx = 0
    temp1_mpy = 0
    temp2_mpy = 0
    temp1_p1 = 0
    temp1_p2 = 0
    temp2_p1 = 0
    temp2_p2 = 0
    right_hand = 0
    left_hand = 0
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            p1,p2 =  detector_utils.draw_box_on_image(i,scores, boxes, im_width, im_height,image)
            mpx =  int((p1[0]+p2[0])/2)
            mpy =  int((p1[1]+p2[1])/2)
            if i == 0:
                temp1_mpx = mpx
                temp1_p1 = p1
                temp1_p2 = p2
                temp1_mpy = mpy
            else:
                temp2_mpx = mpx
                temp2_p1 = p1
                temp2_p2 = p2
                temp2_mpy = mpy
        else:
            temp2_mpx = 105
            temp2_mpy = -10
            im_crop = cv2.resize(blank_image,(60,60))
            right_hand = im_crop
            left_hand = im_crop
    try:
        if temp1_mpx > temp2_mpx :
            blank_image = cv2.circle(blank_image,(temp1_mpx,temp1_mpy),2,(0,0,255),5)
            blank_image = cv2.circle(blank_image,(temp2_mpx,temp2_mpy),2,(0,255,0),5)
            im_crop = image[temp1_p1[1]:temp1_p2[1],temp1_p1[0]:temp1_p2[0]]
            im_crop = cv2.resize(im_crop,(60,60))
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR)
            right_hand = im_crop
            im_crop = image[temp2_p1[1]:temp2_p2[1],temp2_p1[0]:temp2_p2[0]]
            im_crop = cv2.resize(im_crop,(60,60))
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR)
            left_hand = im_crop
            
        else:
            blank_image = cv2.circle(blank_image,(temp2_mpx,temp2_mpy),2,(0,0,255),5)
            blank_image = cv2.circle(blank_image,(temp1_mpx,temp1_mpy),2,(0,255,0),5)
            im_crop = image[temp1_p1[1]:temp1_p2[1],temp1_p1[0]:temp1_p2[0]]
            im_crop = cv2.resize(im_crop,(60,60))
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR)
            left_hand = im_crop
            im_crop = image[temp2_p1[1]:temp2_p2[1],temp2_p1[0]:temp2_p2[0]]
            im_crop = cv2.resize(im_crop,(60,60))
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR)
            right_hand = im_crop
    except:
        print('No hands')   
    
    hcen = cv2.resize(blank_image,(60,60))
    return right_hand,left_hand,hcen

