from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300_new import ssd_300_new
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

import os

def lr_schedule(epoch):
    if epoch < 80:
        return 0.0001
    elif epoch < 100:
        return 0.00001
    else:
        return 0.000001

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales = scales_pascal
scales = [0.147,0.307,0.547,0.76,0.867,0.973,1.0]

# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
aspect_ratios = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]

two_boxes_for_ar1 = False
# steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
steps= None
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.

clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.1, 0.1] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

load_opts = 0 # 0: from scratch; 1: from pretrained weigths; 2: from pretrained model
if load_opts == 0 or load_opts == 1:
    # 1: Build the Keras model.
    K.clear_session() # Clear previous models from memory.
    model, predictor_size = ssd_300_new(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=mean_color,
                    swap_channels=swap_channels,
                    return_predictor_sizes=True)
    print("predictor_size: {}".format(predictor_size))
    print(model.summary())
    
    # 2: Load some weights into the model.
    ############################################################################################
    if load_opts == 1:
        ## TODO: Set the path to the weights you want to load.
        weights_path = 'output/ssd300_snapshots/weights/ssd300_pascal_07+12_epoch-30_loss-14.7925_val_loss-14.6434.h5'
        model.load_weights(weights_path, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

elif load_opts == 2:
    ## Load a previously created model
    ## TODO: Set the path to the `.h5` file of the model to be loaded.
    model_path = 'output/ssd300_new/snapshots/models/ssd300_pascal_07+12_epoch-01_loss-20.8205_val_loss-16.3313.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    K.clear_session() # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                'L2Normalization': L2Normalization,
                                                'compute_loss': ssd_loss.compute_loss})
else:
    print("Unknown load_opts. 0: from scratch; 1: from pretrained weigths; 2: from pretrained model, but got {}.".format(load_opts))

#################################################################################################
# Data generator for training
#################################################################################################
# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.
# TODO: Set the paths to the datasets here.
# The directories that contain the images.
VOC_2007_images_dir      = '../../Data/VOC/VOCdevkit_trainval/VOC2007/JPEGImages/'
VOC_2012_images_dir      = '../../Data/VOC/VOCdevkit_trainval/VOC2012/JPEGImages/'
VOC_2007_test_images_dir = '../../Data/VOC/VOCdevkit_test/VOC2007/JPEGImages'

# The directories that contain the annotations.
VOC_2007_annotations_dir      = '../../Data/VOC/VOCdevkit_trainval/VOC2007/Annotations/'
VOC_2012_annotations_dir      = '../../Data/VOC/VOCdevkit_trainval/VOC2012/Annotations/'
VOC_2007_test_annotations_dir = '../../Data/VOC/VOCdevkit_test/VOC2007/Annotations'

# The paths to the image sets.
VOC_2007_train_image_set_filename    = '../../Data/VOC/VOCdevkit_trainval/VOC2007/ImageSets/Main/train.txt'
VOC_2012_train_image_set_filename    = '../../Data/VOC/VOCdevkit_trainval/VOC2012/ImageSets/Main/train.txt'
VOC_2007_val_image_set_filename      = '../../Data/VOC/VOCdevkit_trainval/VOC2007/ImageSets/Main/val.txt'
VOC_2012_val_image_set_filename      = '../../Data/VOC/VOCdevkit_trainval/VOC2012/ImageSets/Main/val.txt'
VOC_2007_trainval_image_set_filename = '../../Data/VOC/VOCdevkit_trainval/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_filename = '../../Data/VOC/VOCdevkit_trainval/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_filename     = '../../Data/VOC/VOCdevkit_test/VOC2007/ImageSets/Main/test.txt'
# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                     VOC_2012_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                             VOC_2012_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir,
                                          VOC_2012_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[VOC_2007_test_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_test_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.
if not os.path.isfile('dataset_pascal_voc_07+12_trainval.h5'):
    train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)
if not os.path.isfile('dataset_pascal_voc_07_test.h5'):
    val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

######################################################################################################
## Initialize for training
######################################################################################################
# 3: Set the batch size.
batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.
# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)
# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('pool3_mbox_conf').output_shape[1:3],
                   model.get_layer('pool4_mbox_conf').output_shape[1:3],
                   model.get_layer('conv5_2_mbox_conf').output_shape[1:3],
                   model.get_layer('pool5_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3]]
ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)
val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)
# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()
print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))  

# Define model callbacks.
# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='output/ssd300_new/snapshots/models/ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best = 
csv_logger = CSVLogger(filename='output/ssd300_new/logs/ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)
learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)
terminate_on_nan = TerminateOnNaN()
callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

################################################################################################
##  Train
################################################################################################
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 120
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)                          
