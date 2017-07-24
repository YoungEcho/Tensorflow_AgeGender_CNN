# -*- coding: utf-8 -*-

'''
__author__ = 'youngtong'

'This is a python file for Processing Data'
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import glob

age_table = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_table = ['1', '2']  

# AGE==True 
AGE = False

if AGE == True:
    lables_size = len(age_table)  # age
else:
    lables_size = len(sex_table)  # gender

face_set_fold = 'E:\Python36_File\TensorflowLearn/Database/IMDB_WIKI/wiki_crop'

def get_picture( imgpath=face_set_fold, CLASSNUM=lables_size, BatchSize = 128):
    all_data_label = []     
    all_data_set = []  

    num = 0   
    for dirpath, dirnames, filenames in os.walk(imgpath):
        for dirname in dirnames:        
            chilDirFile = os.listdir(os.path.join(dirpath, dirname))
            for File in chilDirFile:
                File = os.path.join(dirpath, dirname, File)
                img = cv2.imread(File,cv2.IMREAD_GRAYSCALE)
                # cv2.imshow('TrainImg',img)
                # cv2.waitKey(1000)
                img = img[15:107, :]         
                # cv2.imshow('TImg',img)
                # cv2.waitKey(0)
                tmparr = np.array(img).flatten()      
                all_data_set = np.hstack( (all_data_set,tmparr) )

                # tmpLabel = np.zeros([1, 40])
                # tmpLabel[:,-1 - num] = 1
                # all_data_label.append( tmpLabel )

                all_data_label.append(num)
            num += 1
    ###### Labels processing to meet the need of requirement of dense Tensor
    # label = tf.expand_dims(tf.reshape(all_data_label,[1,-1]),1)
    label = tf.reshape(all_data_label,[-1,1])

    # index = tf.range(0,label.get_shape().as_list()[-1])
    index = tf.expand_dims(tf.range(0,label.get_shape().as_list()[0]),1)
    # index = tf.range(0,label.get_shape().as_list()[-1])
    concated = tf.concat(1, [index, label])     
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([label.get_shape().as_list()[0], CLASSNUM]), 1.0, 0.0)
    return [np.array(all_data_set,'float32'), onehot_labels]

'''
Accoding txt file to load files path and labels
'''
def parse_data(face_image_set=face_set_fold):
    data_set = []
    fold_x_data = os.path.join(face_set_fold,'gender.txt')

    with open(fold_x_data, 'r',newline='\n') as f:
        line_one = True
        for line in f:
            tmp = []
            if line_one == True:        
                line_one = False
                continue

            tmp.append(line.split('\t')[0])
            tmp.append(line.split('\t')[1][0:-1])       

            file_path = os.path.join(face_image_set, tmp[1])
            if os.path.exists(file_path):
                # filenames = glob.glob(file_path + "/*.jpg")
                # filenames = file_path
                if AGE == True:
                    if tmp[2] in age_table:
                        data_set.append([file_path, age_table.index(tmp[2])])
                else:
                    if tmp[0] in sex_table:
                        data_set.append([file_path, sex_table.index(tmp[0])])
    return data_set
