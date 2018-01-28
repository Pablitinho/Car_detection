## Writeup Car Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car.png
[image2]: ./images/no_car.png
[image3]: ./images/YUV_HOG_car.jpg
[image4]: ./images/YUV_HOG_no_car.jpg
[image5]: ./images/detection_1.jpg
[image6]: ./images/detection_2.jpg
[image7]: ./images/detection_3.jpg
[image8]: ./images/detection_4.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! :)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

At first I created a python script "prepare_data.py" where I collect all the images and generate a pickle file in order to 
be more easy the training part. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

The training script is in the file "training.py" where I read the pickles files with the data of car and no car. Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. First image belong to a car with the HOG features and the other one to a no car image:

![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

At the beginning I used the gray scale image and getting the hog features, but after evaluate it on the video, I got a lot of false positive. At this point I used the three channels and extracting the hog features per channel, but I got the same results... a lot of false positives. After used as input other color spaces I discovered that the YUV color space was working pretty good in the video. Also I downscaled the image to 32x32 to speed up the training and hence the detection.

HoG Parameters: 
orient=9
pix_per_cell=8
cell_per_block=2


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a a SVM machine classifier that is in the file "Classifier.py" file. The code to train is as follow:

def train_svm_classifier(data,label):

    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(data, label)

    pred = clf.predict(data)
    acc = accuracy_score(pred, label)
    print("Training Accuracy: ", acc*100, "%")

    return clf
 
As you can see, I used the GridSearchCV function to find the best kernel and the best parameter. The best result are making use of RBF with C=10 obtaining a 100% of accuracy.
 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The implementation of this part is in the file "main.py" in the function "find_cars()". After do some experiments, I realize that using the 64x64 patches the whole method was working pretty slow and using low scales. So I decided to reduce the patches to 32x32 and use more lower scales (3,4). As overlap parameter I used cells_per_step = 2 because I obtained better result but if I use the cells_per_step = 1 the algorithm was pretty slow.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features  getting very nice results. In order to optimize the performance I used only this 3 Hog features because with those features are enough to get nice results. To get better performance I used the 32x32 image patches and using two small scales (Scale = 3 and Scale = 4). Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

 I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  In order to avoid more the false positives, I included a temporal filter as follow:
 	
     alfa = 0.2	
     heat_temp = heatmap+(heat_temp*(1-alfa)+heatmap*alfa)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

