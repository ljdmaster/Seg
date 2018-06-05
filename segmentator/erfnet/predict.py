from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf

from common.viz import vizseg
from data.voc2012.data_processing import prepare_data, calculate_class_weights
from common.model_base import SegmentationModel
from common.erfnet import erfnetA, erfnetB
from common.image_processing import create_augmentation_func, create_augmentation_func_for_segmentation


'''
aug_func = create_augmentation_func_for_segmentation(
    shadow=(0.01, 0.8),
    shadow_file="./data/voc2012/shadow_pattern.jpg",
    shadow_crop_range=(0.02, 0.5),
    rotate=30,
    crop=0.66,
    lr_flip=False,
    tb_flip=False,
    brightness=(0.5, 0.4, 4),
    contrast=(0.5, 0.3, 5),
    blur=1,
    noise=10
    )
'''


# Visualize samples of augmentations
# from viz import viz_sample_augmentations
# viz_sample_segmentation_augmentations(data["X_train"], data["Y_train"], colormap=data["colormap"], aug_func=aug_func, n_images=2, n_per_image=5, saveto="sample_augmentation_pairs.jpg")

# ##############################################################################
#                                                                           MAIN
# ##############################################################################
if __name__ == '__main__':
    # SETTINGS
    n_valid = 128
    data_file = "./data/voc2012/data_256.pickle"
    # vgg16_snapshot = "/path/to/vgg16/vgg_16.ckpt"
    # vgg16_snapshot = "/home/ronny/TEMP/pretrained_models/tfslim/vgg/vgg16/vgg_16.ckpt"

    # PREPARE DATA
    DATA_LIMIT = None
    data = prepare_data(data_file, valid_from_train=True, n_valid=n_valid, max_data=DATA_LIMIT)
    n_classes = len(data["id2label"])

    # MODEL - ERFNet, with Paszke class weighting
    model_name = "aug_erfnetC_03"
    model = SegmentationModel(model_name, img_shape=[256,256], n_classes=len(data["id2label"]), l2=2e-4)
    
    class_weights = calculate_class_weights(data["Y_train"], n_classes=n_classes, method="paszke", c=1.10)
    model.set_class_weights(class_weights)
    model.create_graph(erfnetB) 
    
    preds = model.predict(data["X_valid"][:9], batch_size=9, best=False, verbose=False)
    output = vizseg(img=data["X_valid"][:9], 
                    label=data["Y_valid"][:9],
                    pred=preds[:9],
                    colormap=data.get("colormap", None),
                    gridshape=[3,3],
                    saveto="predict.jpg")


