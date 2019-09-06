from keras import backend as K
from keras.models import load_model,Model
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import sys

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set the image size.
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales = [0.147,0.307,0.547,0.76,0.867,0.973,1.0] # My scale values are calculated based on perspective zone
aspect_ratios = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]] # Anchorboxes are boundingboxes of perspective zones
steps= None
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.

clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
model_mode = 'inference'

load_opt = 1 # 0: load weight ; 1: load model
if load_opt == 0:
    # 1: Build the Keras model
    print("develop later")
    
    # # TODO: Need to edit below commands
    # K.clear_session() # Clear previous models from memory.
    # model = ssd_300(image_size=(img_height, img_width, 3),
    #                 n_classes=n_classes,
    #                 mode=model_mode,
    #                 l2_regularization=0.0005,
    #                 scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    #                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
    #                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
    #                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
    #                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
    #                                          [1.0, 2.0, 0.5],
    #                                          [1.0, 2.0, 0.5]],
    #                 two_boxes_for_ar1=True,
    #                 steps=[8, 16, 32, 64, 100, 300],
    #                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    #                 clip_boxes=False,
    #                 variances=[0.1, 0.1, 0.2, 0.2],
    #                 normalize_coords=True,
    #                 subtract_mean=[123, 117, 104],
    #                 swap_channels=[2, 1, 0],
    #                 confidence_thresh=0.01,
    #                 iou_threshold=0.45,
    #                 top_k=200,
    #                 nms_max_output_size=400)

    # # 2: Load the trained weights into the model.
    # # TODO: Set the path of the trained weights.
    # weights_path = 'output/ssd300/snapshots/weights/ssd300_pascal_07+12_epoch-78_loss-13.047_val_loss-12.8245.h5'
    # model.load_weights(weights_path, by_name=True)

    # # 3: Compile the model so that Keras won't complain the next time you load it.
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    # model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    # model.summary()
    # # sys.exit()
elif load_opt ==1:
    # from keras_layers.keras_layer_DecodeDetections_V1 import DecodeDetections_V1
    from keras_layers.keras_layer_DecodeDetections import DecodeDetections
    model_path = 'output/ssd300_new_adam_ssd/snapshots/models/ssd300_pascal_07+12_epoch-49_loss-4.8732_val_loss-4.9873.h5'
    
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    K.clear_session() # Clear previous models from memory.
    train_model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                         'L2Normalization': L2Normalization,
                                                         'compute_loss': ssd_loss.compute_loss})
    train_output_layer = "predictions"
    predictions = train_model.get_layer(train_output_layer).output
    decoded_predictions = DecodeDetections(confidence_thresh=0.5,
                                               iou_threshold=0.5,
                                               top_k=200,
                                               nms_max_output_size=400,
                                               coords='centroids',
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
    model = Model(input = train_model.input,
                  output= decoded_predictions)
    print(model.summary())
else:
    print("Do not know load_opt. Expect 0/1 but got {}.".format(load_opt))
############################################################################################################################
############################################################################################################################
# Create data generator

dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = '../../Data/VOC/VOCdevkit_test/VOC2007/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = '../../Data/VOC/VOCdevkit_test/VOC2007/Annotations/'
Pascal_VOC_dataset_image_set_filename = '../../Data/VOC/VOCdevkit_test/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

###################################################################################################
###################################################################################################
# Evaluation
######################
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.1,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results   

###################################################################################################
###################################################################################################
# Visualize result
######################
print(average_precisions)
print("Recall")
# print(recalls)
for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))