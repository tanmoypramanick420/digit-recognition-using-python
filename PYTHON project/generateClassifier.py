from joblib import dump
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv("train.csv")
labels = data["label"]
features = data.drop(['label'], axis=1)
numpy_array = features.to_numpy()

list_hog_fd = []

for feature in numpy_array:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')
print("Count of digits in dataset", Counter(labels))

clf = LinearSVC()

clf.fit(hog_features, labels)

dump(clf, "digits_cls .pkl", compress=3)

print("Classifier Generated....")




