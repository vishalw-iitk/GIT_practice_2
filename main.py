from face_detection import facial_landmarks
from hand_detection import hand_centre
import cv2
import os

import pandas as pd

main_folder = 'Train_data'
save_folder = 'pipeline_train_data'
for folder in os.listdir(main_folder):
    if not os.path.exists(os.path.join(save_folder,folder)):
        os.mkdir(os.path.join(save_folder,folder))
        blank_image = np.zeros((260,210,3), np.uint8)
        temp = np.zeros((60,70,3), np.uint8)
         in os.listdir(os.path.join(main_folder,folder)):
            print(image_path)
            image_str = image_path.split('_')
            os.mkdir(os.path.join(save_folder,folder,image_str[4]))
            image = cv2.imread(os.path.join(main_folder,folder,image_path))
            face_crop = face_landmarks.detect(temp,image)
            face_image_naav = 'face_crop_'+image_path+'_'+'.png'
            cv2.imwrite(os.path.join(save_folder,folder,image_str[4],face_image_name),face_crop)
            blank_image = hand_centre.detect(save_folder,folder,image,blank_image,image_str[4])
        flow_path = 'flow_'+image_path+'_'+'.png'   
        cv2.imwrite(os.path.join(save_folder,folder,flow_path),blank_image)
    else:
        print('folder_exists')



