from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing import image
from PIL import Image as pil_img
from keras.optimizers import Adam,SGD
from imageio import imread, imwrite
import numpy as np
from matplotlib import pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('../'))
from ssd_encoder_decoder.ssd_input_encoder_V1 import SSDInputEncoder_V1
from keras_layers.keras_layer_DecodeDetections_V1 import DecodeDetections_V1
# from keras_encoder_decoder.ssd_output_decoder_V1 import 

img_height = 300
img_width = 300
n_classes = 2
global_pos_iou_threshold = 0.3
# # one predictor layer
# predictor_sizes = [[2,2]]
# scales=[0.5,1.0]
# aspect_ratios_per_layer=[[1.0]]
# offsets=[0.5]

# n predictor layers
predictor_sizes = [[4,4],[3,3], [2,2]]
scales=[0.25, 0.33, 0.5, 1.0]
aspect_ratios_per_layer=[[1.0],[1.0],[1.0]]
offsets=[0.5,0.5,0.5]
normalize_coords = True

I = np.ones((img_height,img_width)).astype("uint8")*100
# I = pil_img.from_array(np.ones((img_height,img_width)).astype("uint8")*100)
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

ssd_input_encoder = SSDInputEncoder_V1(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios_per_layer,
                                    two_boxes_for_ar1=False,
                                    steps=None,
                                    offsets=offsets,
                                    clip_boxes=False,
                                    variances=[0.1, 0.1, 0.1, 0.1],
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    global_pos_iou_threshold=global_pos_iou_threshold,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)
groundtruth_labels = [np.array([[1,50,50,100,100],
                                [2,150,150,210,230]]).reshape(2,5)]                                
encoded_input = ssd_input_encoder(groundtruth_labels)
print(encoded_input)
print(type(encoded_input))
print(encoded_input.shape)

show_dummy = True
if show_dummy:
    offset = 0
    for pred_size in predictor_sizes:
        plt.figure(figsize=(20,12))
        plt.imshow(I)
        current_axis = plt.gca()
        current_axis.set_xticks(np.arange(0,img_width,img_width/pred_size[1]))
        current_axis.set_yticks(np.arange(0,img_height,img_height/pred_size[0]))
        anchor_start_idx = offset
        anchor_end_idx = offset + pred_size[0]*pred_size[1] - 1        
        offset = anchor_end_idx + 1        
        for gt_idx, gt in enumerate(groundtruth_labels):
            for r_idx in range(gt.shape[0]):
                color = colors[int(gt[r_idx,0])]
                current_axis.add_patch(plt.Rectangle((gt[r_idx,1], gt[r_idx,2]), gt[r_idx,3]-gt[r_idx,1], gt[r_idx,4]-gt[r_idx,2], color=color, fill=False, linewidth=2))  
        
            anchors = encoded_input[gt_idx,anchor_start_idx:anchor_end_idx,:]
            # convert to absolute coordinate
            if normalize_coords:
                anchors[:,[-8,-6]] *= img_width
                anchors[:,[-7,-5]] *= img_height
            # get the best anchor boxes
            confident_score = np.copy(anchors[:,1:n_classes+1])
            max_conf = np.max(confident_score,axis=-1)
            for a_idx in range(anchors.shape[0]):
                if max_conf[a_idx]>global_pos_iou_threshold:
                    color = colors[10]
                    # it is for centroid format
                    current_axis.add_patch(plt.Rectangle((anchors[a_idx,-8]-int(anchors[a_idx,-6]/2), 
                                                        anchors[a_idx,-7]-int(anchors[a_idx,-5]/2)), 
                                                        anchors[a_idx,-6], anchors[a_idx,-5], color=color, fill=False, linewidth=2,linestyle="--"))

        plt.grid()
        plt.show()

# K.clear_session()





