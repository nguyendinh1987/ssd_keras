from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Input, Lambda
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import os

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

import cv2
print("Load libraries: done")
print("Assigned tensor dimension")
print(K.image_data_format())
def identity_layer(tensor):
    return tensor

# Read image:
img_path = "examples/fish-bike.jpg"
I = cv2.imread(img_path)
img_h, img_w, img_c = I.shape

# Create network that has only AnchorBoxes layer for testing
input = Input(shape=(img_h,img_w,img_c))
boxes4 = Lambda(identity_layer, output_shape=(img_h, img_w, img_c), name='identity_layer')(input)
anchorB = AnchorBoxes(img_h*2, img_w*2, this_scale=0.5, next_scale=0.25, aspect_ratios=[1,0.5,2],
                           two_boxes_for_ar1=True, this_steps=None, this_offsets=None,
                           clip_boxes=True, variances=np.array([1.0,1.0,1.0,1.0]), coords="centroids", normalize_coords=False, name='anchorB')(boxes4)
model = Model(inputs=input,outputs=anchorB)
print("Model summary:")
print(model.summary())

# Get output
dummy = model.predict(np.expand_dims(I,0))
print("output type")
print(type(dummy))
print("output shape:")
print(dummy.shape)

# get boxes at specific pixels
# get boxes at cx and cy
cx = 50
cy = 50
Fboxes = dummy[:,cy,cx,:,:]
print(Fboxes)

# Show image
cv2.imshow("x",I)
cv2.waitKey(0)
cv2.destroyAllWindows()