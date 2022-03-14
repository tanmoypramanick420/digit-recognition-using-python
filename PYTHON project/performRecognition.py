import cv2
from joblib import load
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = load("digits_cls .pkl")

# Read the input image
im = cv2.imread("photo_2.jpg")

# Convert to grayscale (from colored image to only gray image)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter (to reduce noise in the image or downgrade the image)
im_gray = cv2.GaussianBlur (im_gray, (5, 5), 0)

# Threshold the image (Basically convert that Gray image to black and white image)
ret, im_th= cv2.threshold (im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image (find the white area or patterned area)
ctrs, hier = cv2.findContours (im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]


for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))

    # Doing prediction using the classifier
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()



