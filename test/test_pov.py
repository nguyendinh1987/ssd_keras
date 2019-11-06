from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
import cv2


import sys
import os
sys.path.append(os.path.abspath('../'))
from models.keras_ssd300_pview import ssd_300_pview
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder_pview import SSDInputEncoder_Pview
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy_pview, match_multi

img_height = 300
img_width = 300
img_channels = 3
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
n_classes = 20
scales = [0.147,0.28,0.567,0.78,1,1.1]
aspect_ratios = [[1.0],[1.0],[1.0],[1.0],[1.0]]
two_boxes_for_ar1 = False 
steps = [8,16,16,32,32]
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5]
offsets = [3.5, 7.5,23.5, 55.5,119.5]
clip_boxes = False 
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = False
coords='corners'

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

K.clear_session() # Clear previous models from memory.
model = ssd_300_pview(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='inference',
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
                        swap_channels=swap_channels)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)                        
print(model.summary())

I = cv2.imread("../examples/000453.jpg")
I = cv2.resize(I,(img_width,img_height), interpolation = cv2.INTER_AREA)

y_predict = model.predict(np.expand_dims(I,axis=0))
print(y_predict.shape)
cv2.imshow('image',I)
cv2.waitKey(0)
cv2.destroyAllWindows()