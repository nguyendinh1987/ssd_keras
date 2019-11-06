import sys
import os
sys.path.append(os.path.abspath('../'))
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy_pview, match_multi,clip_boxes
from bounding_box_utils.bounding_box_utils import iou_V1

import numpy as np
from matplotlib import pyplot as plt
# test for 3 classes
img_h = 300
img_w = 300
xmin = 1
ymin = 2
xmax = 3
ymax = 4
# each row is [xmin, ymin, xmax, ymax] --> corners format
anchors = np.array([[0        ,0        ,img_w    ,img_h    ], # cell 1x1
                    [0        ,0        ,img_w/2  ,img_h/2  ], # cell 2x2
                    [img_w/2  ,0        ,img_w    ,img_h/2  ],
                    [0        ,img_h/2  ,img_w/2  ,img_h    ],
                    [img_w/2  ,img_h/2  ,img_w    ,img_h    ],
                    [0        ,0        ,img_w/3  ,img_h/3  ], # cell 3x3 
                    [img_w/3  ,0        ,img_w*2/3,img_h/3  ],
                    [img_w*2/3,0        ,img_w    ,img_h/3  ],
                    [0        ,img_h/3  ,img_w/3  ,img_h*2/3],
                    [img_w/3  ,img_h/3  ,img_w*2/3,img_h*2/3],
                    [img_w*2/3,img_h/3  ,img_w    ,img_h*2/3],
                    [0        ,img_h*2/3,img_w/3  ,img_h    ],
                    [img_w/3  ,img_h*2/3,img_w*2/3,img_h    ],
                    [img_w*2/3,img_h*2/3,img_w    ,img_h    ]]) 

# each row is [class_id, xmin, ymin, xmax, ymax] --> corners format
labels = np.array([[3,img_w/8  ,img_h/8   ,img_w*3/8  ,img_h*5/8  ], 
                   [1,img_w*5/8,img_h/16  ,img_w*7/8  ,img_h*5/16 ],
                   [2,img_w*7/16,img_h*7/16,img_w*15/16,img_h*14/16],
                   [3,0        ,img_h*11/16,img_w/2    ,img_h      ]])
# visualize
I = np.ones((img_h,img_w)).astype("uint8")*100
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.figure(figsize=(20,12))
plt.imshow(I)
current_axis = plt.gca()
anchor_color = colors[0]
for a_idx in range(anchors.shape[0]):
    current_axis.add_patch(plt.Rectangle((anchors[a_idx,0], anchors[a_idx,1]),anchors[a_idx,2]-anchors[a_idx,0]+1,anchors[a_idx,3]-anchors[a_idx,1]+1,
                           color=anchor_color, fill=False, linewidth=2,linestyle="--"))
gt_color = colors[2]
for a_idx in range(labels.shape[0]):
    current_axis.add_patch(plt.Rectangle((labels[a_idx,1], labels[a_idx,2]),labels[a_idx,3]-labels[a_idx,1]+1,labels[a_idx,4]-labels[a_idx,2]+1,
                           color=gt_color, fill=False, linewidth=2,linestyle="--"))
plt.show()

n_class = 3
classes = np.eye(n_class+1)
labels_one_hot = np.concatenate([classes[labels[:,0].astype(np.int),:],labels[:, [xmin,ymin,xmax,ymax]]],axis=1)
y_encoded = np.concatenate([np.zeros([anchors.shape[0],n_class+1]),
                           anchors,
                           anchors,
                           anchors],axis=1)
y_encoded[:,0] = 1                           

iou2gt = iou_V1(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[:,-12:-8], coords="corners", mode='outer_product', border_pixels="half",iou_type="to_box1")
iou = iou_V1(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[:,-12:-8], coords="corners", mode='outer_product', border_pixels="half")
print("---------------------iou2gt-----------------------------------")
print(iou2gt)
print("----------------------iou-------------------------------------")
print(iou)
print("--------------------------------------------------------------")

print("test match_bipartite_greedy_pview")
m = match_bipartite_greedy_pview(iou,iou2gt,pos_threshold = 0.2)
print(m)
need_delete_indices = np.where(m==-1)
m_clean = np.delete(m,need_delete_indices,0)
labels_one_hot_cleaned = np.delete(labels_one_hot,need_delete_indices,0)
print(m_clean)
print("labels_one_hot")
print(labels_one_hot)
print("labels_one_hot_cleaned")
print(labels_one_hot_cleaned)
print("----------------------y_encoded-------------------------------")
print(y_encoded)
y_encoded[m_clean, :-8] = labels_one_hot_cleaned
print("----------------------y_encoded bipartite---------------------")
print(y_encoded)
print("")
print("")
print("")
print("test multi matches")
iou[:,m_clean] = 0
iou2gt[:,m_clean] = 0
matches = match_multi(weight_matrix=iou, threshold=0.5)
gts_indices_matches = matches[0]
ans_indices_matches = matches[1]
print("gts_indices_matches")
print(gts_indices_matches)
print("ans_indices_matches")
print(ans_indices_matches)
labels_scores = labels_one_hot[gts_indices_matches]
one_hot_indices = np.where(labels_scores[:,:n_class+1]==1)
labels_scores[one_hot_indices[0],one_hot_indices[1]] = iou2gt[gts_indices_matches,ans_indices_matches]
print("labels_scores")
print(labels_scores)
y_encoded[ans_indices_matches,:-8] = labels_scores
print("y_encoded")
print(y_encoded)

print("")
print("")
print("")
print("test clip gt")
gtboxes = y_encoded[:,-12:-8]
anchorboxes = y_encoded[:,-8:-4]
print(gtboxes.shape)
print(anchorboxes.shape)
# anchor_color = colors[0]
# gt_color = colors[2]
# for a_idx in range(anchorboxes.shape[0]):
#     plt.figure(figsize=(20,12))
#     plt.imshow(I)
#     current_axis = plt.gca()
#     current_axis.add_patch(plt.Rectangle((anchorboxes[a_idx,0], anchorboxes[a_idx,1]),
#                                           anchorboxes[a_idx,2]-anchorboxes[a_idx,0]+1,
#                                           anchorboxes[a_idx,3]-anchorboxes[a_idx,1]+1,
#                                           color=anchor_color, fill=False, linewidth=2,linestyle="--"))
#     current_axis.add_patch(plt.Rectangle((gtboxes[a_idx,0], gtboxes[a_idx,1]),
#                                           gtboxes[a_idx,2]-gtboxes[a_idx,0]+1,
#                                           gtboxes[a_idx,3]-gtboxes[a_idx,1]+1,
#                                           color=gt_color, fill=False, linewidth=2,linestyle="--"))
#     plt.show()
gtboxes = clip_boxes(gtboxes,anchorboxes)
for a_idx in range(anchorboxes.shape[0]):
    plt.figure(figsize=(20,12))
    plt.imshow(I)
    current_axis = plt.gca()
    current_axis.add_patch(plt.Rectangle((anchorboxes[a_idx,0], anchorboxes[a_idx,1]),
                                          anchorboxes[a_idx,2]-anchorboxes[a_idx,0]+1,
                                          anchorboxes[a_idx,3]-anchorboxes[a_idx,1]+1,
                                          color=anchor_color, fill=False, linewidth=2,linestyle="--"))
    current_axis.add_patch(plt.Rectangle((gtboxes[a_idx,0], gtboxes[a_idx,1]),
                                          gtboxes[a_idx,2]-gtboxes[a_idx,0]+1,
                                          gtboxes[a_idx,3]-gtboxes[a_idx,1]+1,
                                          color=gt_color, fill=False, linewidth=2,linestyle="--"))
    plt.show()




