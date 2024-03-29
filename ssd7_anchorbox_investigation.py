from keras import backend as K
#set specific GPU
config = K.tf.ConfigProto(device_count={'GPU':0,'GPU':1, 'GPU':2})
sess = K.tf.Session(config=config)
####################################################
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread, imwrite
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from bounding_box_utils.bounding_box_utils import convert_coordinates, locate_feature_area, cell_boundingbox, img_generation


print("Load libraries: done")

# Set the image size.
img_height = 300
img_width = 480
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 5 # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


# Build keras model
# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)
print(model.summary())
load_opt = 1
if load_opt == 1: # Load trained weights into model (designed model is maintained)
    # TODO: Set the path of the trained weights.
    epoch = 18
    loss = 2.1442
    val_loss = 2.3166
    weights_path = 'output/snapshots/weights/ssd7_epoch-{}_loss-{}_val_loss-{}.h5'.format(epoch,loss,val_loss)
    model.load_weights(weights_path, by_name=True)
    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
else: # Load pretrained model (designed model could be changed by loaded model)
    # TODO: Set the path to the `.h5` file of the model to be loaded.
    epoch = 15
    loss = 2.1735
    val_loss = 2.2954
    model_path = 'output/snapshots/ssd7_epoch-{}_loss-{}_val_loss-{}.h5'.format(epoch,loss,val_loss)
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    #K.clear_session() # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
print("Load model weights: done")
# Read images
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.
# We'll only load one image in this example.
img_path = 'examples/000453.jpg'#new_image_0.jpg'#
orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

##########################################################################
print("Try to get output tensors at given layers")
layer_name_0 = "predictions"
layer_name_1 = "decoded_predictions"
output_anchors4 = "anchors4"
output_anchors5 = "anchors5"
output_anchors6 = "anchors6"
output_anchors7 = "anchors7"
injected_model = Model(inputs = model.input,
                       outputs=[model.get_layer(layer_name_0).output,
                                model.get_layer(layer_name_1).output,
                                model.get_layer(output_anchors4).output,
                                model.get_layer(output_anchors5).output,
                                model.get_layer(output_anchors6).output,
                                model.get_layer(output_anchors7).output])
injected_outputs = injected_model.predict(input_images)
anchors_range = [0]
print('output shape')
for op_idx, op in enumerate(injected_outputs):
    print(op.shape)
    if op_idx > 1:
        anchors_range.append(op.shape[1]*op.shape[2]*4-1+anchors_range[op_idx-2])
print(anchors_range)                       
predictions = injected_outputs[0]
y_pred = injected_outputs[1]
# anchorboxes4 = injected_outputs[2]
# anchorboxes5 = injected_outputs[3]
# anchorboxes6 = injected_outputs[4]
# anchorboxes7 = injected_outputs[5]
anchor_list = injected_outputs[2:6]
for an in anchor_list:
    print(an.shape)
##########################################################################

print('y_pred shape')
print(y_pred.shape)
confidence_threshold = 0.6
y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])
print(y_pred_thresh[0].shape)

# Visualization
# Display the image and draw the predicted boxes onto it.
# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'car', 'truck', 'pedestrian', 'bicyclist', 'traffic light']

plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box_idx, box in enumerate(y_pred_thresh[0]):
    # Get anchor box
    anchor_id = box[-9]
    cx = box[-8]* orig_images[0].shape[1] / img_width
    cy = box[-7]* orig_images[0].shape[0] / img_height
    cw = box[-6]* orig_images[0].shape[1] / img_width
    ch = box[-5]* orig_images[0].shape[0] / img_height
    anchorbox = np.array([cx,cy,cw,ch])
    # because anchorbox in centroids mode so:
    anchorbox = convert_coordinates(anchorbox, start_index=0, conversion='centroids2corners', border_pixels='half')
    grid_size, cell_id = locate_feature_area(anchor_list,anchor_id)
    print("grid size")
    print(grid_size)
    cell_box = cell_boundingbox(orig_images[0].shape,grid_size,cell_id)
    print("cell box")
    print(cell_box)

    # create new image
    new_image = np.zeros(orig_images[0].shape)
    v_offset = 3*(cell_box[2] - cell_box[0])
    h_offset = 3*(cell_box[3] - cell_box[1])
    crop_y0 = max(0,int(cell_box[1] - h_offset/2))
    crop_y1 = min(orig_images[0].shape[0],int(cell_box[3] + h_offset/2))
    crop_x0 = max(0,int(cell_box[0] - v_offset/2))
    crop_x1 = min(orig_images[0].shape[1],int(cell_box[2] + v_offset/2))
    cropped_img = np.copy(orig_images[0][crop_y0:crop_y1,crop_x0:crop_x1,:])
    new_image = new_image + img_generation(np.mean(orig_images[0]),np.std(orig_images[0]),[new_image.shape[0],new_image.shape[1]])
    new_image[crop_y0:crop_y1,crop_x0:crop_x1,:] = cropped_img
    imwrite('examples/new_image_'+str(box_idx)+'.jpg',new_image)
    # Transform the predicted bounding boxes for the 300x480 image to the original image dimensions.
    xmin = box[-4] * orig_images[0].shape[1] / img_width
    ymin = box[-3] * orig_images[0].shape[0] / img_height
    xmax = box[-2] * orig_images[0].shape[1] / img_width
    ymax = box[-1] * orig_images[0].shape[0] / img_height
    print([xmin,ymin,xmax,ymax])
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    print(label)
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label+"_"+str(box_idx), size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    # draw anchorbox
    current_axis.add_patch(plt.Rectangle((anchorbox[0], anchorbox[1]), anchorbox[2]-anchorbox[0], anchorbox[3]-anchorbox[1], color=colors[7], fill=False, linewidth=2))  
    current_axis.text(anchorbox[0], anchorbox[1], "_anchor", size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    # draw feature region
    current_axis.add_patch(plt.Rectangle((cell_box[0], cell_box[1]), cell_box[2]-cell_box[0], cell_box[3]-cell_box[1], color=colors[8], fill=False, linewidth=1))  
    current_axis.text(cell_box[0], cell_box[1], "_feature_zone", size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
plt.show()

# Crop feature region and create a new image
# Running net predict on new image
