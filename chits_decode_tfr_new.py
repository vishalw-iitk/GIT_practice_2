# Demonstration of extracting TFRecord file with images stored in Bytes

import tensorflow as tf
import os
import shutil
import matplotlib.image as mpimg
import numpy as np
import cv2

class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        print("#######",self.tfrecord_file)

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        features = {
            'len': tf.FixedLenFeature([], tf.int64),
        }
        sample = tf.parse_single_example(tfrecord, features)
        leng = sample['len']
        #leng = tf.decode_raw(sample['len'], tf.string)
        #leng = tf.reshape(leng, [1])


        features = {
            '0': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'len': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
        }
        print("$$$$$$$$$$$$$$$",features)

        # Extract the data record
        sample = tf.parse_single_example(tfrecord, features)
        #print("@@@@@@@@",sample)
        image = sample['0']
        image = tf.decode_raw(image, tf.float32)
        image = tf.reshape(image, [60, 60, 4])
        #image = tf.image.decode_image(sample['0'])
        print(image)
        #img_shape = tf.stack([sample['rows'], sample['cols'], sample['channels']])
        label = sample['label']
        label = tf.decode_raw(sample['label'], tf.float32)
        label = tf.reshape(label, [1,100])
        h = sample['height']
        #leng = sample['len']
        w = sample['width']
        
        print("******",image,"*****",label,"***",leng,"****",h,"****",w)
        return [image, label]
        #return [image, label, filename, img_shape]        

    def extract_image(self):
        # Create folder to store extracted images
        folder_path = 'ExtractedImages_ImgAsBytes'
        #shutil.rmtree(folder_path, ignore_errors = True)
        #os.mkdir(folder_path)

        # Pipeline of dataset and iterator 
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])

        dataset = dataset.map(self._extract_fn)
        print("+++++++++++++++",dataset)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("&&&&&&&&&&")
            #print("&&&&&&&&&&")
            # try:
            # Keep extracting data till TFRecord is exhausted
            while True:
                image_data = sess.run(next_image_data)
                print(type(image_data[0]))
                img_to_save = image_data[0][:,:,1].reshape(60,60,1)
                #img_to_save = image_data[0][0][0]
                #print("^^^^^^",img_to_save.shape)
                print(type(img_to_save))
                print(img_to_save)
                print("&&&&&&&&&&",img_to_save*255)
                #cv2.imshow("image",255*img_to_save)
                cv2.imwrite("/home/vishal/Desktop/decoded_image.jpg",255*img_to_save)
                #cv2.destroyAllWindows()

                # Check if image shape is same after decoding
                if not np.array_equal(image_data[0].shape, image_data[3]):
                    print('Image {} not decoded properly'.format(image_data[2]))
                    continue
                    
                save_path = os.path.abspath(os.path.join(folder_path, image_data[2].decode('utf-8')))
                mpimg.imsave(save_path, image_data[0])
                print('Save path = ', save_path, ', Label = ', image_data[1])
            # except:
            #     pass

if __name__ == '__main__':
    t = TFRecordExtractor('/home/vishal/Desktop/tf_data_path/Apache_beam_records_-00007-of-00020')
    t.extract_image()