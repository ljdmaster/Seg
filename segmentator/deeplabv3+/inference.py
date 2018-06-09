"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

import deeplab_model
from deeplab_model import deeplab_v3_plus_generator
from utils import preprocessing
from utils import dataset_util
import cv2


def batch_predict(**params):
  # Model
  model = tf.estimator.Estimator(
      model_fn=deeplab_model.deeplabv3_plus_model_fn,
      model_dir=params["model_dir"],
      params={
          'output_stride': params["output_stride"],
          'batch_size': 1,  # Batch size must be 1 because the images' size may differ
          'base_architecture': params["base_architecture"],
          'pre_trained_model': None,
          'batch_norm_decay': None,
          'num_classes': params["num_classes"]
      })
   
  # Example
  image_names = os.listdir(params["data_dir"])
  image_files = [os.path.join(params["data_dir"], image_name) for image_name in image_names]
  
  # Predict
  predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn(image_files),
        hooks=None)
  
  # Output
  output_dir = params["output_dir"]
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  for pred, image_file in zip(predictions, image_files):
    label = pred['classes'][:,:,0]
    
    # Show Label
    print("{} label shape:{} , type: {}".format(image_file, label.shape, label.dtype)) 
    cv2.imshow("label", np.uint8(label*30))
    cv2.waitKey(1000) 





class Inference(object):
  def __init__(self, height, width, **params ):
    self.model_dir = params["model_dir"]
    self.height = height
    self.width = width
    self.params = params
    self.build() 
    self.init()  

  def build(self):
    ## model
    self.Y = tf.placeholder(tf.int32, [None, self.height, self.width,1])
    self.X = tf.placeholder(tf.float32, [None, self.height, self.width, 3])
    self.net = deeplab_v3_plus_generator(self.params["num_classes"],
                                         self.params["output_stride"],
                                         self.params["base_architecture"],
                                         self.params["pre_trained_model"],
                                         self.params["batch_norm_decay"])
    self.logits = self.net(self.X, False)
    self.pred_classes = tf.expand_dims(tf.argmax(self.logits, axis=3, output_type=tf.int32), axis=3)
  
  def init(self):
    ## initialize
    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)
    self.saver = tf.train.Saver()
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
    
  def predict(self, image_files): 
    """ image file list """ 
    labels = []
    # Read Image
    for i in range(len(image_files)):
      images,_ = preprocessing.eval_input_fn([image_files[i]])  
      images_ = self.sess.run(images)
      pred_classes_ = self.sess.run(self.pred_classes, feed_dict={self.X:images_})
      
      label = pred_classes_[0,:,:,0]
      labels.append(label)
      print("{} label shape:{} , type: {}".format(image_files[i], label.shape, label.dtype)) 
       
      # Show result
      decode_labels = preprocessing.decode_labels(pred_classes_, 1, params['num_classes'])
      #cv2.imshow("label", np.uint8(label*30))
      cv2.imshow("decode_label", decode_labels[0])
      cv2.waitKey(1000)

    return labels 


if __name__=="__main__":
  ## hyperparameter
  params = {"output_stride":16,
            "batch_size":1,
            "base_architecture":"resnet_v2_101",
            "pre_trained_model": None,
            "batch_norm_decay": None,
            "num_classes": 7,   #!!
            "model_dir": "./model/train",
            "data_dir": "./data/temp/data_dataset_voc/JPEGImages",
            "output_dir": "./data/temp/output"
           }

  #batch_predict(**params)
    
  height= None
  width = None
  infer = Inference(height, width, **params)
  image_names = os.listdir(params['data_dir']) 
  image_files = [os.path.join(params['data_dir'], image_name) for image_name in image_names]
  infer.predict(image_files)

    
