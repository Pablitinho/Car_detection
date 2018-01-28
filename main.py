import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from global_functions import *
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import imageio as imio
import time

#dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
# load the model from disk
filename = 'svm_model.sav'
svc = pickle.load(open(filename, 'rb'))
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]

filename = 'Scaler.pick'
X_scaler = pickle.load(open(filename, 'rb'))

svc = pickle.load(open("svm_model_hog_3_YUV.sav", 'rb'))
orient=9
pix_per_cell=8
cell_per_block=2
spatial_size = (32, 32)
hist_bins = 32

img = mpimg.imread('./data/test_images/test4.jpg')
#---------------------------------------------------------------------------------------------------------------------------
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
#---------------------------------------------------------------------------------------------------------------------------
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
#---------------------------------------------------------------------------------------------------------------------------
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
#----------------------------------------------------------------------------------------------------------------------------
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.float32(np.copy(img))
    #img = img.astype(np.float32) / 255
    img = convert_color(img, conv='RGB2YUV')
    img_tosearch = img[ystart:ystop, :, :]
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    #ctrans_tosearch = convert_color(img_tosearch, conv='BGR2GRAY')
    ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    #ch1 = ctrans_tosearch
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 32
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            #hog_features = hog_feat1
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            s_img=ctrans_tosearch[ytop:ytop + window, xleft:xleft + window]
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (32, 32))

            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((hog_features, hist_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            #test_features = hog_features
            #test_features=test_features.reshape(1, -1)
            #test_features = np.squeeze(test_features, 1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box_list.append(((xbox_left, ytop_draw + ystart), \
                                 (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img, box_list
#---------------------------------------------------------------------------------------------------------------------------
def multi_scale_detection(img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins):

    #-----------------------------------------------------------------------------------------------------------------------------
    # scale = 5.0
    #
    # start = time.time()
    #
    # out_img_0, boxes_0 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                      hist_bins)
    # end = time.time()
    # print((end - start) * 1000)
    #-----------------------------------------------------------------------------------------------------------------------------
    scale = 4.0

    start = time.time()

    out_img_1, boxes_1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                         hist_bins)
    end = time.time()
    print((end - start) * 1000)
    #-----------------------------------------------------------------------------------------------------------------------------

    # scale = 2.0
    # start = time.time()
    #
    # out_img_2, boxes_2 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    #                     hist_bins)
    # end = time.time()
    # print((end - start) * 1000)
    #-----------------------------------------------------------------------------------------------------------------------------

    scale = 3.0
    start = time.time()

    out_img_3, boxes_3 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    end = time.time()
    print((end - start)*1000)
    #-----------------------------------------------------------------------------------------------------------------------------

    box_list = []

    #box_list.extend(boxes_0)
    box_list.extend(boxes_1)
    #box_list.extend(boxes_2)
    box_list.extend(boxes_3)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 0)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img_heat = draw_labeled_bboxes(np.copy(img), labels)

    return labels, draw_img_heat, heatmap, box_list
# -----------------------------------------------------------------------------------------
ystart = 400
ystop = 656

first_time = True
cap = cv2.VideoCapture('project_video.mp4')
if (cap.isOpened()== False):
  print("Error opening video file")
  exit(-1)

size = (int(1280), int(720))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'x264' doesn't work
out_video_heat = cv2.VideoWriter('./output_heat.avi', fourcc, 20.0, size, True)  # 'False' for 1-ch instead of 3-ch for color
out_video_detection = cv2.VideoWriter('./out_video_detection.avi', fourcc, 20.0, size, True)  # 'False' for 1-ch instead of 3-ch for color

heat_temp = 0

for i in range(8*30):
    ret, image = cap.read()

print("Start the show....")
while cap.isOpened():

    # Read images
    ret, image = cap.read()

    labels, draw_img_heat, heatmap, box_list = multi_scale_detection(image, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)


    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)


    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img_heat = draw_labeled_bboxes(np.copy(image), labels)

    if first_time:
       first_time = False
       heat_temp = heatmap

    alfa = 0.2
    heat_temp = heatmap+(heat_temp*(1-alfa)+heatmap*alfa)

    # Apply threshold to help remove false positives
    heat_temp = apply_threshold(heat_temp, 1)

    heat_temp[(heat_temp > 255)] = 255

    heat_visualization = (heat_temp*10)
    heat_visualization[(heat_visualization > 255)] = 255
    heat_visualization = np.uint8(heat_visualization)

    print("Maximum:", np.max(heat_visualization))

    heat_visualization = np.uint8(heat_visualization)
    heat_visualization = np.expand_dims(heat_visualization, axis=2)
    heat_visualization = cv2.cvtColor(heat_visualization, cv2.COLOR_GRAY2RGB)

    cv2.imshow("Visualization", draw_img_heat)
    cv2.waitKey(20)

    #cv2.imshow("Heat", heat_visualization)
    #cv2.waitKey(20)
    out_video_heat.write(heat_visualization)
    out_video_detection.write(draw_img_heat)

# plt.imshow(out_img_0)
# plt.show()
#
# plt.imshow(out_img_1)
# plt.show()
#
# plt.imshow(out_img_2)
# plt.show()
