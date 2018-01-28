import classifier as classifier
import pickle
import numpy as np
from global_functions import *
import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage.feature import hog
# -------------------------------------------------------------------------------------------------------------
def load_data(path,file_name):
    with open(path + file_name + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data
# -------------------------------------------------------------------------------------------------------------

orient=9
pix_per_cell=8
cell_per_block=2
hist_bins = 32

X_scaler = StandardScaler()

X_scaler.copy = True
X_scaler.with_mean = True
X_scaler.with_std = True

filename = 'svm_model.sav'

data_vehicle = load_data("./data/", "Vehicle")
data_no_vehicle = load_data("./data/", "No_Vehicle")

ones = np.ones(len(data_vehicle))
zeros = np.zeros(len(data_no_vehicle))
feature_list = [ones, zeros]

label = np.hstack(feature_list)
# for idx in range(len(data_no_vehicle)):
#     img = data_no_vehicle[idx]
#     img_YUV = convert_color(img, conv='RGB2YUV')
#     hog_feat1,img1 = get_hog_features(img_YUV[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
#     hog_feat2,img2 = get_hog_features(img_YUV[:, :, 1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
#     hog_feat3,img3 = get_hog_features(img_YUV[:, :, 2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
#
#     f, axarr = plt.subplots(1,4)
#     axarr[0].imshow(img)
#     axarr[1].imshow(img1)
#     axarr[2].imshow(img2)
#     axarr[3].imshow(img3)
#     plt.show()
#     pp = 0

feature_list = []
for idx in range(len(data_vehicle)):
    img = data_vehicle[idx]
    img = cv2.resize(img, (32, 32))
    #img = img.astype(np.float32) / 255
    img_YUV = convert_color(img, conv='RGB2YUV')
    #hist_features = color_hist(img_YCRCB, nbins=hist_bins)
    hog_feat1 = get_hog_features(img_YUV[:, :, 0], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_feat2 = get_hog_features(img_YUV[:, :, 1], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_feat3 = get_hog_features(img_YUV[:, :, 2], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    #features = np.hstack((hog_feat1, hog_feat2, hog_feat3, hist_features))
    features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    feature_list.append(features)

    # print(features)
    # cv2.imshow("hog", hog_image)
    # cv2.waitKey(0)
for idx in range(len(data_no_vehicle)):
    img = data_no_vehicle[idx]
    img = cv2.resize(img, (32, 32))
    #img = img.astype(np.float32) / 255
    img_YUV = convert_color(img, conv='RGB2YUV')
    #hist_features = color_hist(img_YCRCB, nbins=hist_bins)
    hog_feat1 = get_hog_features(img_YUV[:, :, 0], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_feat2 = get_hog_features(img_YUV[:, :, 1], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    hog_feat3 = get_hog_features(img_YUV[:, :, 2], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    #features = np.hstack((hog_feat1, hog_feat2, hog_feat3,hist_features))
    features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
    feature_list.append(features)

print("Features generated.....")
X = np.vstack(feature_list).astype(np.float64)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
feature_list = X_scaler.transform(X)
pickle.dump(X_scaler, open("Scaler.pick", 'wb'))
data = feature_list

clf = classifier.train_svm_classifier(data, label)

print("System Trained.....")

# load the model from disk
#clf = pickle.load(open(filename, 'rb'))

pred = classifier.evaluate_svm(clf, data)

# Save the model to disk
filename = 'svm_model_hog_3_YUV.sav'
pickle.dump(clf, open(filename, 'wb'))

print("Model Saved.....")

print(pred)
