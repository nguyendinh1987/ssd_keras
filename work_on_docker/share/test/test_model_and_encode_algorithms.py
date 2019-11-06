from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt


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

predictor_sizes = [model.get_layer('conf_conv4_3').output_shape[1:3],
                   model.get_layer('conf_conv5_3').output_shape[1:3],
                   model.get_layer('conf_conv6_1').output_shape[1:3],
                   model.get_layer('conf_conv7_2').output_shape[1:3],
                   model.get_layer('conf_conv9_2').output_shape[1:3]]
print(predictor_sizes)                   
ssd_input_encoder = SSDInputEncoder_Pview(img_height=img_height,
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
                                    coords = coords,
                                    pos_iou_threshold=0.7, # Consider only anchorbox cover almost gt
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords,
                                    pos_threshold = 0)

ground_truth_labels = [np.array([[3,img_width/8   ,img_height/8    ,img_width*3/8  ,img_height*5/8  ], 
                                 [1,img_width*5/8 ,img_height/16   ,img_width*7/8  ,img_height*5/16 ],
                                 [2,img_width*7/16,img_height*7/16 ,img_width*15/16,img_height*14/16],
                                 [3,0             ,img_height*11/16,img_width/2    ,img_height      ]])]
y_encode = ssd_input_encoder(ground_truth_labels)
if True:
    # draw anchor boxes and gt boxes
    an_boxes = y_encode[0,:,-8:-4]
    print(an_boxes[:20,:])
    I = np.ones((img_height,img_width,3)).astype("uint8")*100
    anchor_color = colors[0]
    gt_color = colors[2]
    danchor_color = colors[3]
    for a_idx in range(an_boxes.shape[0]):
        plt.figure("Fig1",figsize=(20,12))
        plt.imshow(I)
        current_axis = plt.gca()
        labels = ground_truth_labels[0]
        for gt_idx in range(labels.shape[0]):
            current_axis.add_patch(plt.Rectangle((labels[gt_idx,1], labels[gt_idx,2]),labels[gt_idx,3]-labels[gt_idx,1]+1,labels[gt_idx,4]-labels[gt_idx,2]+1,
                            color=gt_color, fill=False, linewidth=2,linestyle="--"))
        if np.sum(y_encode[0,a_idx,:-12],axis=-1) == 0 or y_encode[0,a_idx,0] == 1:
            current_axis.add_patch(plt.Rectangle((an_boxes[a_idx,0], an_boxes[a_idx,1]),an_boxes[a_idx,2]-an_boxes[a_idx,0]+1,an_boxes[a_idx,3]-an_boxes[a_idx,1]+1,
                                color=anchor_color, fill=False, linewidth=2,linestyle="--"))
            plt.pause(0.05)                                
        else:
            current_axis.add_patch(plt.Rectangle((an_boxes[a_idx,0], an_boxes[a_idx,1]),an_boxes[a_idx,2]-an_boxes[a_idx,0]+1,an_boxes[a_idx,3]-an_boxes[a_idx,1]+1,
                                color=danchor_color, fill=False, linewidth=5,linestyle="--"))
            plt.pause(0.5)
        plt.clf()
    plt.show()

y_decode = np.copy(y_encode[:,:,:-8])
if coords == 'corners':
    y_decode[:,:,-4:] *= y_encode[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
    y_decode[:,:,[-4,-2]] *= np.expand_dims(y_encode[:,:,-6] - y_encode[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
    y_decode[:,:,[-3,-1]] *= np.expand_dims(y_encode[:,:,-5] - y_encode[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
    y_decode[:,:,-4:] += y_encode[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    if normalize_coords:
        y_decode[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_decode[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates
else:
    raise ValueError("coords must be corners")

neutral_boxes = np.where(np.sum(y_encode[:,:,:-12],axis=-1)==0)
print("Number of neutral boxes is {}".format(len(neutral_boxes[1])))
y_decode = np.delete(y_decode,neutral_boxes[1],1)
y_encode = np.delete(y_encode,neutral_boxes[1],1)
bg_boxes = np.where(y_decode[:,:,0]==1)
print("Number of background boxes is {}".format(len(bg_boxes[1])))
y_decode = np.delete(y_decode,bg_boxes[1],1)
y_encode = np.delete(y_encode,bg_boxes[1],1)

gt_boxes = y_decode[0,:,-4:]
an_boxes = y_encode[0,:,-8:-4]
print("groundtruth boxes")
print(gt_boxes.shape)
print(gt_boxes)
print("anchor boxes")
print(an_boxes.shape)
print(an_boxes)

if True:
    I = np.ones((img_height,img_width,3)).astype("uint8")*100
    anchor_color = colors[0]
    gt_color = colors[2]
    plt.figure("Fig1",figsize=(20,12))
    plt.imshow(I)
    current_axis = plt.gca()
    labels = gt_boxes
    for gt_idx in range(labels.shape[0]):
        current_axis.add_patch(plt.Rectangle((labels[gt_idx,0], labels[gt_idx,1]),labels[gt_idx,2]-labels[gt_idx,0]+1,labels[gt_idx,3]-labels[gt_idx,1]+1,
                        color=gt_color, fill=False, linewidth=2,linestyle="--"))
    for a_idx in range(an_boxes.shape[0]):
        current_axis.add_patch(plt.Rectangle((an_boxes[a_idx,0], an_boxes[a_idx,1]),an_boxes[a_idx,2]-an_boxes[a_idx,0]+1,an_boxes[a_idx,3]-an_boxes[a_idx,1]+1,
                            color=anchor_color, fill=False, linewidth=1,linestyle="--"))
    plt.show()

