import sys
import os
sys.path.append(os.path.abspath('../'))
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy_pview
from bounding_box_utils.bounding_box_utils import iou_V1

import numpy as np
# test for 3 classes
img_h = 300
img_w = 300
# each row is [xmin, ymin, xmax, ymax] --> corners format
anchors = [[0      ,0       ,img_w/2,img_h/2], # cell 2x2
           [img_w/2,0       ,img_w  ,img_h/2],
           [0      ,img_h/2 ,img_w/2,img_h  ],
           [img_w/2,img_h/2 ,img_w  ,img_h  ]] 
gts = [[]] # each row is [class_id, xmin, ymin, xmax, ymax] --> corners format
label_one_hot = np.eye(3)


iou2gt = np.array([[0.3,  1,0.2,  1,0.21],
                   [0.3,0.9,0.5,0.2,0.67],
                   [  1,0.1,0.4,  1,0.95],
                   [0.5,  1,0.7,  1,0.13]])

iou = np.array([[0.33,0.65,0.28,0.75,0.26],
                [0.36,0.94,0.53,0.25,0.64],
                [0.66,0.16,0.43,0.48,0.23],
                [0.52,0.21,0.72,0.94,0.15]])


m = match_bipartite_greedy_pview(iou,iou2gt,pos_threshold = 0.5)
print(m)
need_delete_indices = np.where(m==-1)
m = np.delete(m,need_delete_indices,0)
print(m)


