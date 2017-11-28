import sys
sys.path.append('/libsvm-3.22/python') #Path to libsvm directory
from PIL import Image
from sklearn import svm
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

start_time = time.time()

if multiprocessing.cpu_count() > 2:
    cores = multiprocessing.cpu_count()-2
else:
    cores = 1

ROOT_DIR = '/Images/' #Path to the Images directory

#Store data and labels
ALL_TRAIN_IMAGES = []
ALL_TRAIN_LABELS = []


# Loading the file that has the names of the classes we will do the test on. For testing purposes, this file can be only a subset of all the breeds.
# For actual running, the file with the full list of breeds can be chosen.

#f = open ('/all_breed_names', 'r') # Path to file containing all the breeds
f = open ('/limited_breed_names', 'r') # Path to file containing only limited breeds for testing

lines = f.readlines()

# Create list to store all the classes and two dictionaries to translate a breed name to a unique number, and vice versa.
class_list = []
number_to_class_convert = {}
class_to_number_convert = {}
number = 1
for line in lines:
    class_list.append(line.lstrip().rstrip())
    number_to_class_convert[number] = line.lstrip().rstrip()
    class_to_number_convert[line.lstrip().rstrip()] = number
    number = number+1


time_to_start_image_convert = time.time()
# Test for any image files that may not be in the expected format. Ignore any such file.
incorrect_file = []
for c in class_list:
    print 'Going to start converting the class - '+c
    list_of_images = os.listdir(ROOT_DIR+c)
    for image in list_of_images:
        #print 'converting the file - '+ROOT_DIR+c+'/'+image+'\n'

        # Converting each image to numpy array using ANTIALIAS filter for good quality. Resizing the pixels.
        image_object = Image.open(ROOT_DIR+c+'/'+image).resize((50,50),Image.ANTIALIAS)
        a = np.array(image_object)

        if a.shape == (50,50, 3):
            ALL_TRAIN_IMAGES.append(a)
            ALL_TRAIN_LABELS.append(class_to_number_convert[c])
        else:
            print 'INCORRECT FILE FOUND! - '+c+'/'+image
            incorrect_file.append(c+'/'+image)
print 'TOTAL TIME TO CONVERT ALL IMAGES AND LABELS = %s SECONDS' %(time.time() - time_to_start_image_convert) 


# Convert the lists to numpy arrays and reshape
ALL_TRAIN_LABELS_NP = np.asarray(ALL_TRAIN_LABELS)
ALL_TRAIN_IMAGES_NP = np.asarray(ALL_TRAIN_IMAGES)

n_samples = len(ALL_TRAIN_IMAGES_NP)
data = ALL_TRAIN_IMAGES_NP.reshape((n_samples, -1))
labels = ALL_TRAIN_LABELS_NP


# Normalizing the data.
# It did not help with the score.
#normalized_data = normalize(data)


# Feature Reduction Attempts
# 1. PCA
#pca=PCA(3000)
#data_pca_transformed = pca.fit_transform(data)
#data_pca_transformed = pca.fit_transform(normalized_data)
#print data_pca_transformed.shape
#print 'PCA variance ratio' + str(pca.explained_variance_ratio_.sum())

# 2. TruncatedSVD
#svd = TruncatedSVD(1500)
#data_svd_transformed = svd.fit_transform(data)
#data_svd_transformed = svd.fit_transform(normalized_data)
#print 'SVD variance ratio' + str(svd.explained_variance_ratio_.sum())


# Split the data to training and testing data.
print '\n Calculating the randomly generated training and testing data and labels set \n'
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=0.80, random_state=42)
#data_train, data_test, labels_train, labels_test = train_test_split(normalized_data, labels, train_size=0.95, random_state=42)
#pca_data_train, pca_data_test, pca_labels_train, pca_labels_test = train_test_split(data_pca_transformed, labels, train_size=0.80, random_state=42)
#svd_data_train, svd_data_test, svd_labels_train, svd_labels_test = train_test_split(data_svd_transformed, labels, train_size=0.80, random_state=42)
print 'Datasets generated'


# Try different Models and calculate results.

# 1. Linear SVC
print '\nTrying to fit Linear SVC classifier'
lin_clf = LinearSVC()
lin_clf.fit(data_train, labels_train)
print 'Score:'
print lin_clf.score(data_test, labels_test)

# 2. SVC
print '\nTrying to fit SVC classifier'
svc_clf = SVC()
svc_clf.fit(data_train, labels_train)
print 'Score:'
print svc_clf.score(data_test, labels_test)

# 3. Random Forests
print '\nTrying to fit Random Forsest classifier'
rf_clf = RandomForestClassifier(min_samples_leaf=20, n_jobs=cores)
rf_clf.fit(data_train, labels_train)
print 'Score:'
print rf_clf.score(data_test, labels_test)

# 4. kNN
print '\nTrying to fit kNN classifier'
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(data_train, labels_train)
print 'Score:'
print knn.score(data_test, labels_test)

# 5. OneVsRest classifier - THIS IS CRASHING WHEN RUNNING IN IDLE. So need to test from terminal.
#print '\nTrying to fit OneVsRest classifer'
#ovr_clf = OneVsRestClassifier(LinearSVC(), n_jobs = cores)
#ovr_clf.fit(data_train, labels_train)
#print 'Score:'
#print ovr_clf.score(data_test, labels_test)

print '\n TOTAL TIME TO FOR ENTIRE FILE TO COMPLETE = %s SECONDS' %(time.time() - start_time) 

