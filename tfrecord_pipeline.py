import pandas as pd
import numpy as np
from cv2 import cv2
from face_detection import face_landmarks
from hand_detection import hand_centre
import argparse
import os
import logging
import gcsfs 
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io import tfrecordio
import apache_beam as beam

LABEL_DICT = {'A LOT OF': 0, 'AIR': 1, 'ALSO': 2, 'AUGUST': 3, 'BEAUTIFUL': 4, 'BEGINNING': 5, 'BUT': 6, 'CAN': 7, 'CHANGE': 8, 'CLEAR': 9, 'CLOUD': 10, 'COLD': 11, 'COME': 12, 'COURSE': 13, 'DAMP': 14, 'DEEP': 15, 'DIFFERENTLY': 16, 'DRY': 17, 'EIGHT': 18, 'EIGHTEEN': 19, 'ELF': 20, 'EQUAL': 21, 'ESPECIALLY': 22, 'EVE': 23, 'FIFTEEN': 24, 'FIRST TIME': 25, 'FRESH': 26, 'FRIDAY': 27, 'FRIENDLY': 28, 'GERMAN': 29, 'GOOD': 30, 'HIGH': 31, 'HOT': 32, 'IF': 33, 'IN-COMING': 34, 'JUST': 35, 'KUESTE': 36, 'LABOR': 37, 'LANG': 38, 'LIKE': 39, 'LITTLE': 40, 'LUNCH': 41, 'MASS': 42, 'MAXIMUM': 43, 'MINUS': 44, 'MONDAY': 45, 'MORE': 46, 'MORNING': 47, 'MOSTLY': 48, 'MOUNTAIN': 49, 'NINE': 50, 'NORTH': 51, 'NORTHEAST': 52, 'NORTHWEST': 53, 'NOT': 54, 'NOW': 55, 'OTHERWISE': 56, 'PART': 57, 'PLACE': 58, 'PRINT': 59, 'PROBABLY': 60, 'Purchase': 61, 'RAIN': 62, 'RARE': 63, 'RIVER': 64, 'SATURDAY': 65, 'SEE': 66, 'SEVEN': 67, 'SIX': 68, 'SIXTEEN': 69, 'SKY': 70, 'SNOW': 71, 'SOMETIMES': 72, 'STARK': 73, 'SUED': 74, 'SUEDOST': 75, 'SUN': 76, 'SUNDAY': 77, 'SWEAT': 78, 'TAG': 79, 'TEMPERATURE': 80, 'TEN': 81, 'THEN': 82, 'THERE': 83, 'THIRTY': 84, 'THREE': 85, 'THUNDERSTORM': 86, 'THURSDAY': 87, 'TO': 88, 'TODAY': 89, 'TUESDAY': 90, 'TWENTY': 91, 'TWO': 92, 'WARM': 93, 'WEAK': 94, 'WEATHER': 95, 'WEDNESDAY': 96, 'WEST': 97, 'WIND': 98, '[COULD]': 99}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-path',
        type=str,
        default='gs://qommunicator/CMLE_Pipeline/Test_words_full.csv',
        help='name of csv file')
    return parser.parse_known_args()
 




class LoadImageDoFn(beam.DoFn):

    def __init__(self,data_csv):
        self.data_csv = pd.read_csv(data_csv)


    def process(self, text_line):
        print(text_line)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        labels = self.data_csv['labels']
        image_folder_paths = self.data_csv['folder_path']
        fs = gcsfs.GCSFileSystem(project='speedy-aurora-193605',access='full_control')
        print("fs==",fs.ls)
        print("before loop1")
        for label,path in zip(labels,image_folder_paths):
            image_arr = []
	        print("after loop1")
            print("path",path)
            print("before loop2",fs.ls(path))
            print("length", len(fs.ls(path)))
            for image in fs.ls(path):
		        #print("image_numer",i)
                print("after loop2")
                #image = "gs://"+image
                print("orig_path",image)                
		        #print(image)
                with fs.open(image) as img_file:
                    img = np.array(load_img(img_file,target_size=(260,210)))
                    # print(os.path.join(path,image))
                    # print(img.shape)
                    image_arr.append(img)
                    # print(label)
                    # print(len(image_arr))
            label = LABEL_DICT[label]
            label = tf.keras.utils.to_categorical(label, 100)
            yield (image_arr,label)
    

            
class PreprocessImagesDoFn(beam.DoFn):
    def process(self,image_label_tuple):
        images,label = image_label_tuple
        # print(len(images))
        # print(label)
        for i in range(len(images)):
            image = images[i]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detected_face = face_landmarks.detect(image)
            hright,hleft,hcen = hand_centre.detect(image)
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            hright = cv2.cvtColor(hright, cv2.COLOR_BGR2GRAY)
            hleft = cv2.cvtColor(hleft, cv2.COLOR_BGR2GRAY)
            hcen = cv2.cvtColor(hcen, cv2.COLOR_BGR2GRAY)
            print(detected_face.shape,hright.shape,hleft.shape,hcen.shape)
            stack = np.dstack((detected_face,hright,hleft,hcen)) 
            stack = np.asarray(stack)
            stack = stack/255
            images[i] = stack
        yield (images,label)

       

class ImageToTfExampleDoFn(beam.DoFn):

    def __init__(self):
        print("Running")
    
    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def process(self,image_label_tuple):
        images, label = image_label_tuple
        feature = {}
        g_labels = label.astype(np.float32)
        feature['label'] = self._bytes_feature(g_labels.tostring())
        for index in range(len(images)):
            path = str(index)
            image = images[index].astype('float32')
            image_raw = image.tostring()
            feature[path] = self._bytes_feature(image_raw)
            feature['height'] = self._int64_feature(60)
            feature['width'] = self._int64_feature(60)
            feature['depth'] = self._int64_feature(4)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
	print("feature dictionary",feature)        
	yield example





def run_pipeline():
    args, pipeline_args = get_args()
    # print(pipeline_args)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    read_textline_from_csv = beam.io.ReadFromText(args.csv_path)

    load_img_from_path = LoadImageDoFn(data_csv=args.csv_path)

    augment_data = PreprocessImagesDoFn()

    img_to_tfexample = ImageToTfExampleDoFn()

    write_to_tf_record = tfrecordio.WriteToTFRecord(file_path_prefix='gs://qommunicator/Apache_beam_records/',num_shards=10)

    print("###########################################")
    with beam.Pipeline(options=pipeline_options) as pipe:
        _ = (pipe
             | 'ReadCSVFromText' >> read_textline_from_csv
             | 'LoadImageData' >> beam.ParDo(load_img_from_path)
             | 'PreprocessImages' >> beam.ParDo(augment_data)
             | 'ImageToTfExample' >> beam.ParDo(img_to_tfexample)
             | 'SerializeProto' >> beam.Map(lambda x: x.SerializeToString())
             | 'WriteTfRecord' >> write_to_tf_record)
        print('Done running')
        
def main():
    run_pipeline()



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    main()
    
