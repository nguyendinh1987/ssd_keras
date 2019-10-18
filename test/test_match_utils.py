import sys
import os
sys.path.append(os.path.abspath('../'))
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy_pview

import numpy as np
weight_matrix1 = np.array([[0.3,  1,0.2,  1,0.21],
                           [0.3,0.9,0.5,0.2,0.67],
                           [  1,0.1,0.4,  1,0.95],
                           [0.5,  1,0.7,  1,0.13],
                           [0.3,0.2,  1,0.5,0.82]])

weight_matrix2 = np.array([[0.33,0.65,0.28,0.75,0.26],
                           [0.36,0.94,0.53,0.25,0.64],
                           [0.66,0.16,0.43,0.48,0.23],
                           [0.52,0.21,0.72,0.94,0.15],
                           [0.73,0.42,0.163,0.45,0.52]])
m = match_bipartite_greedy_pview(weight_matrix2,weight_matrix1,pos_threshold = 0.5)
print(m)
need_delete_indices = np.where(m==-1)
print(need_delete_indices)
m = np.delete(m,need_delete_indices,0)
print(m)