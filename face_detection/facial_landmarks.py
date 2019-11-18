"""facial_landmarks.py: To detect facial landmarks of the images of given dataset."""

import os
from mtcnn.mtcnn import MTCNN
from cv2 import cv2
from imutils import face_utils
import dlib
import pandas as pd

def bbox_pred(detections, dat_file):
    """ Predicted the shape of the face as a box and changed the box size too.

    Args:
        why such argumentsngular bounding box on the detected face.

    Return:
        rect: it returns the parameters of the predicted rectangle box of face.
        predictor: it returns the shape of the predicted image.
    """
    predictor = dlib.shape_predictor(dat_file)
    bbox = detections[0]['box']
    x_bbox = bbox[0]-10
    y_bbox = bbox[1]-5
    height_bbox = bbox[2]+20
    width_bbox = bbox[3]+15
    rect = dlib.rectangle(left=x_bbox, top=y_bbox, right=x_bbox+height_bbox, \
        bottom=y_bbox+width_bbox)
    return rect, predictor

def face_detection(data_set, detector, dat_file):
    """ To extract the facial landmarks of the detected face from the image with \
        the corrsponding names in a dataframe.

    Args:
        data_set: it is the folder containing folders of frames of the signs performed.
        detector: it is the MTCNN model used to predict the faces in an image.
        dat_file: used to get the 68 facial landmarks on the detected face.

    Returns:
        dframe: returns the dataframe containing the facial landmarks coordinates \
            in a list of list format with their corresponding name of the images in another list.
    """
    names = []
    landmarks_list = []
    for image_folder in os.listdir(data_set):
        for image in os.listdir(os.path.join(data_set, image_folder)):
            image_list = image.split('_')
            if 'crop.png' not in image_list:
                img = cv2.imread(os.path.join(data_set, image_folder, image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(img)
                if detections:
                    rect, predictor = bbox_pred(detections, dat_file)
                    landmarks = predictor(img, rect)
                    landmarks = list(face_utils.shape_to_np(landmarks))
                    landmarks_list.append(landmarks)
                    names.append(image)
                else:
                    landmarks_list.append(0)
                    names.append(image)
    dframe = pd.DataFrame(list(zip(names, landmarks_list)), columns=['Image_Name', 'Face_landmark'])
    return dframe


if __name__ == "__main__":
    dataset = 'Data_100'
    dat = 'gs://qommunicator/CMLE_Pipeline/shape_predictor_68_face_landmarks.dat'
    face_detection(dataset, MTCNN(), dat).to_csv(r'Labels.csv')



new addition of the line
