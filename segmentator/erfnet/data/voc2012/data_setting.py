#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/18 

import numpy as np

id2label = [ 'void',        # 0
             'aeroplane',   # 1
             'bicycle',     # 2 
             'bird',        # 3 
             'boat',        # 4 
             'bottle',      # 5 
             'bus',         # 6 
             'car',         # 7  
             'cat',         # 8 
             'chair',       # 9 
             'cow',         # 10 
             'diningtable', # 11 
             'dog',         # 12 
             'horse',       # 13 
             'motorbike',   # 14
             'person',      # 15 
             'potted plant',# 16 
             'sheep',       # 17 
             'sofa',        # 18 
             'train',       # 19
             'tv/monitor',  # 20
            ]

# NOTE: RGB channels
label_colormap = { id2label[0]: (  0,   0,   0),
                   id2label[1]: (128,   0,   0),
                   id2label[2]: (  0, 128,   0),
                   id2label[3]: (128, 128,   0),
                   id2label[4]: (  0,   0, 128),
                   id2label[5]: (128,   0, 128),
                   id2label[6]: (  0, 128, 128),
                   id2label[7]: (128, 128, 128),
                   id2label[8]: ( 64,   0,   0),
                   id2label[9]: (192,   0,   0),
                   id2label[10]: ( 64, 128,   0),
                   id2label[11]: (192, 128,   0),
                   id2label[12]: ( 64,   0, 128),
                   id2label[13]: (192,   0, 128),
                   id2label[14]: ( 64, 128, 128),
                   id2label[15]: (192, 128, 128),
                   id2label[16]: (  0,  64,   0),
                   id2label[17]: (128,  64,   0),
                   id2label[18]: (  0, 192,   0),
                   id2label[19]: (128, 192,   0),
                   id2label[20]: (  0,  64, 128)
                }

label2id = {label:id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

Data_path = './'
inputs_subdir = "./train_inputs"
labels_subdir = "./train_labels"
pickle_file = "data_256.pickle"
shape = [256,256]


setting = {"id2label":id2label,
           "label2id":label2id,
           "idcolormap":idcolormap,
           "Data_path":Data_path,
           "inputs_subdir":inputs_subdir,
           "labels_subdir":labels_subdir,
           "pickle_file":pickle_file,
           "shape":shape}

