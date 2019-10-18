'''
Utilities to match ground truth boxes to anchor boxes.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np

def match_bipartite_greedy_pview(weight_matrix_1, weight_matrix_2, pos_threshold=0):
    '''
    - The weight_matrix_1 is offical IoU between two lists of boxes.
    - The weight_matrix_2 is variant IoU between two lists of boxes. It stands 
      a case of pview cover gt bounding box
    - This function is to deal with a case that there are a groundtruth box are 
    covered by many anchor boxes. It will try to select the smallest anchor box.
    -------------------
    - The output is list of anchorboxes that cover gt and have largest IoU to gt
      and their IoU > pos_thrshold (in default, pos_threshold = 0). The order fo-
      -llow gt order. Gt that do not have suitable anchor boxes will be marked as
      [-1].
    '''
    weight_matrix = np.copy(weight_matrix_1) # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int) 

    # clean IoU < pos_threshold
    print(weight_matrix)
    weight_matrix[np.where(weight_matrix <= pos_threshold)] = 0
    print("after")
    print(weight_matrix)
    # Find gt_indices and anchor_indices point to pairs that anchor box covers gt box
    gt_indices, anchor_indices = np.where(weight_matrix_2==1)
    overlaps = weight_matrix[gt_indices,anchor_indices]
    for cur_gt_index in range(num_ground_truth_boxes):
        sl_indices = np.where(gt_indices == cur_gt_index)[0]
        if len(sl_indices) == 0:
            matches[cur_gt_index] = -1
        else:
            overlaps_p_gt = overlaps[sl_indices]

            if np.max(overlaps_p_gt) == 0:
                matches[cur_gt_index] = -1
            else:
                idx= np.argmax(overlaps_p_gt)
                matches[cur_gt_index] = anchor_indices[sl_indices[idx]]

    return matches


def match_bipartite_greedy_V1(weight_matrix_1, weight_matrix_2):
    '''
    - The weight_matrix_1 is variant IoU between two lists of boxes. It repsects to 
    area covered by one of two boxes
    - The weight_matrix_2 is offical IoU between two lists of boxes.
    - This function is to deal with a case that there are a groundtruth box are 
    covered by many anchor boxes. It will try to select the smallest anchor box.
    '''
    weight_matrix = np.copy(weight_matrix_1) # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int) 

    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1) # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        
        # get max value
        max_value = weight_matrix[ground_truth_index,anchor_index]
        gt_indices, anchor_indices = np.where(weight_matrix==max_value)
      
        # len(gt_indices) = len(anchor_indices) can be more than 1, 
        # if len(gt_indices) > 1:
        #   using weight_matrix_2 to clean
        # else:
        #   keep original resuls
        if len(gt_indices) > 1:
            overlaps_2 = weight_matrix_2[gt_indices,anchor_indices]
            max_index_2 = np.argmax(overlaps_2)
            ground_truth_index = gt_indices[max_index_2]
            anchor_index = anchor_indices[max_index_2]
                
        matches[ground_truth_index] = anchor_index # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches

def match_bipartite_greedy(weight_matrix):
    '''
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes
    and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any
    anchor box will be matched first, then out of the remaining ground
    truth boxes, the ground truth box that has the greatest similarity
    with any of the remaining anchor boxes will be matched second, and
    so on. That is, the ground truth boxes will be matched in descending
    order by maximum similarity with any of the respectively remaining
    anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of
    ground truth boxes and `n` is the number of anchor boxes.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''

    weight_matrix = np.copy(weight_matrix) # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):

        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1) # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches

def match_multi_V1(weight_matrix_1, weight_matrix_2, threshold):
    '''
    Give my explaination here
    '''
    num_anchor_boxes = weight_matrix_1.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.    

    # Find the best ground truth match for every anchor box.
    gt_indices_thresh_met = []
    anchor_indices_thresh_met = []
    for anchor_indice in all_anchor_indices:
        tmp = weight_matrix_1[:,anchor_indice]
        
        if np.max(tmp)<threshold:
            continue
        
        anchor_indices_thresh_met.append(anchor_indice)
        
        # [0] because len(tuple)=1
        gt_indices = np.where(tmp==np.max(tmp))[0] 
        if gt_indices.shape[0]>1:
            tmp_2 = weight_matrix_2[gt_indices,anchor_indice]
            gt_indice = gt_indices[np.argmax(tmp_2)]
            gt_indices_thresh_met.append(gt_indice)
        else:
            # [0] because we process column by column so the numpy array has only one element
            gt_indices_thresh_met.append(gt_indices[0])
        
    return np.array(gt_indices_thresh_met), np.array(anchor_indices_thresh_met)



def match_multi(weight_matrix, threshold):
    '''
    Matches all elements along the second axis of `weight_matrix` to their best
    matches along the first axis subject to the constraint that the weight of a match
    must be greater than or equal to `threshold` in order to produce a match.

    If the weight matrix contains elements that should be ignored, the row or column
    representing the respective elemet should be set to a value below `threshold`.

    Arguments:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.
        threshold (float): A float that represents the threshold (i.e. lower bound)
            that must be met by a pair of elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    '''

    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes)) # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0) # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices] # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met
