# -*- coding: utf-8 -*-
'''
 This file provide function to collect tweets based on Stream API

 @author:        "Sibelius Seraphini","Larissa Teixeira"
 @contact:       "sseraphini@albany.edu","larissateixeira92@gmail.com"
 @version:       "1"
'''
import cv2 # OpenCV
import numpy as np # Numpy
from scipy.spatial.distance import pdist # Pairwise distance
from scipy.cluster.hierarchy import linkage, fcluster # hierarchical agglomerative cluster
import sys # argv
import matplotlib.pyplot as plt # plot image
from itertools import permutations # generate all combinations of clusters
import time # To measure time performance

# Parameter for SURF
HESSIAN_THRESHOLD = 400

# List of Keypoint Detectors and Keypoint Descriptors
KP_DET_DESC = { \
    'SIFT'  : cv2.SIFT(), \
    'SURF'  : cv2.SURF(HESSIAN_THRESHOLD), \
    'STAR'  : cv2.FeatureDetector_create("STAR"), \
    'ORB'   : cv2.ORB(),\
    'BRIEF' : cv2.DescriptorExtractor_create("BRIEF"), \
    'BRISK' : cv2.BRISK(), \
    'FREAK' : cv2.DescriptorExtractor_create("FREAK")
    }

def unique_rows(arr):
    ''' Return unique rows of array a '''
    return  np.unique(arr.view(np.dtype((np.void, \
                arr.dtype.itemsize*arr.shape[1])))) \
                .view(arr.dtype).reshape(-1, arr.shape[1])

def show_color_image(img):
    ''' OpenCV load image in BGR, matplotlib uses RGB '''
    #b,g,r = cv2.split(img)
    #img2 = cv2.merge([r,g,b])
    #plt.imshow(img2)

    img_rgb = img[:, :, ::-1]

    plt.imshow(img_rgb)

def load_dict(filename):
    ''' Load a python dictionary from a file '''
    temp = {}

    with open(filename,'r') as f:
        for line in f:
            (key , val) = line.lower().rstrip('\n').split(',')
            temp[key] = val

    return temp

def detect_keypoints(img, method):
    ''' Detect the keypoint in an image with a given method '''
    kp_det = KP_DET_DESC[method]
#    t0 = time.time()
    kp = kp_det.detect(img)
#    t1 = time.time()
#    print('Detect Keypoints: %f s' % (t1-t0))
    return kp

def compute_descriptors(img, kp, method):
    ''' Compute the descriptors for the keypoints with a given method '''
    descriptor = KP_DET_DESC[method]
#    t0 = time.time()
    kp, des = descriptor.compute(img, kp)
#    t1 = time.time()
#    print('Compute Descriptors: %f s' % (t1-t0))
    return kp, des

def match_feature(kp, des, norm):
    ''' Match features given the norm type'''
#    t0 = time.time()

    # BFMatcher with default params
    # cv2.NORM_L2 - SIFT, SURF
    # cv2.NORM_HAMMING - ORB, BRIEF, BRISK
    #bf = cv2.BFMatcher(cv2.NORM_L2)
    bf = cv2.BFMatcher(norm)

    # Calculate matches between a keypoint and k=3 more close keypoints
    # The first match is invalid, because is going to be the same Keypoint
    matches = bf.knnMatch(des, des, k=10)

    # We apply the Lowe and Amerini method to select good matches
    ratio = 0.5
    mkp1, mkp2 = [], []
    for m in matches:
        j = 1
        while(m[j].distance < ratio * m[j+1].distance):
            j = j + 1

        for k in range(1, j):
            temp = m[k]

            # Check if the keypoints are spatial separated
            if pdist( np.array([kp[temp.queryIdx].pt, \
                    kp[temp.trainIdx].pt]) ) > 10:
                mkp1.append( kp[temp.queryIdx] )
                mkp2.append( kp[temp.trainIdx] )

    p1 = np.float32([kp1.pt for kp1 in mkp1])
    p2 = np.float32([kp2.pt for kp2 in mkp2])

    if len(p1) != 0:
        # Remove non-unique pairs of points
        p = np.hstack((p1, p2))
        p = unique_rows(p)
        p1 = np.float32(p[:, 0:2])
        p2 = np.float32(p[:, 2:4])

#    t1 = time.time()
#    print('Match Features: %f s' % (t1-t0))

    return p1, p2


def hierarchical_clustering(p, metric, th):
    ''' Compute the Hierarchical Agglomerative Cluster '''
    distance_p = pdist(p)
    Z = linkage(distance_p, metric)
    C = fcluster(Z, th, 'inconsistent', 4)

    return C

def compute_transformations(C, p, p1, min_cluster_pts):
    ''' Consider the cluster information,
        compute the number of transformations '''
    num_gt = 0
    c_max = np.max(C) # number of cluster
    if c_max > 1:
        for k, j in permutations(range(1, c_max+1), 2):
            z1 = []
            z2 = []
            for r in range(1,p1.shape[0]):
                if (C[r] == k) and (C[r + p1.shape[0]] == j):
                    z1.append(p[r, :])
                    z2.append(p[r+p1.shape[0], :])
                if (C[r] == j) and (C[r + p1.shape[0]] == k):
                    z1.append(p[r+p1.shape[0], :])
                    z2.append(p[r, :])

            z1 = np.array(z1)
            z2 = np.array(z2)

            if (len(z1) > min_cluster_pts) and (len(z2) > min_cluster_pts):
                M, _ = cv2.findHomography(z1, z2, cv2.RANSAC, 5.0)
                if len(M) != 0:
                    num_gt = num_gt + 1
    return num_gt

def plot_image(img, p1, p2, C):
    ''' Plot the image with keypoints and theirs matches '''
    plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.scatter(p1[:, 0],p1[:, 1], c=C[0:p1.shape[0]], s=30)

    for (x1, y1), (x2, y2) in zip(p1, p2):
        plt.plot([x1, x2],[y1, y2], 'c')

    plt.show()

def process_image(img, kp_detector, \
        kp_descriptor, metric, th, min_cluster_pts, plot):
    ''' Process one image with a given
        Keypoint Detector and Keypoint Descriptor '''
    # Detect Keypoints and Compute Descriptors
    kp = detect_keypoints(img, kp_detector)
    kp, des = compute_descriptors(img, kp, kp_descriptor)

    # cv2.NORM_L2 - SIFT, SURF
    # cv2.NORM_HAMMING - ORB, BRIEF, BRISK

    if (kp_descriptor == 'SIFT') or (kp_descriptor == 'SURF'):
        norm = cv2.NORM_L2
    else:
        norm = cv2.NORM_HAMMING

    # Match features
    p1, p2 = match_feature(kp, des, norm)

    # No matches - no geometric transformations - no tampering
    if len(p1) == 0:
        return 0
    else:
        p = np.vstack((p1, p2))

        # Hierarchical Agglomerative Clustering
        C = hierarchical_clustering(p, metric, th)

        # Compute number of transformations
        num_gt = compute_transformations(C, p, p1, min_cluster_pts)

        if plot == True:
            plot_image(img, p1, p2, C)

        return num_gt

def generic_experiment(kp_detector, kp_descriptor):
    ''' Run a generic experiment '''
    # Dataset
    DB = 'MICC-F220'
    DB_DIR = 'database'
    FILE_GROUND_TRUTH = 'groundtruthDB_220.txt'

    # Parameters
    METRIC = 'single' # 'centroid' and 'ward' is not supported in scipy yet
    TH = 2.2
    MIN_CLUSTER_PTS = 4

    # Load ground truth information
    ground = load_dict(DB_DIR + '/' + DB + '/' + FILE_GROUND_TRUTH)

    TP = 0;   # True Positive
    TN = 0;   # True Negative
    FP = 0;   # False Positive
    FN = 0;   # False Negative

    # Apply the method for all images
    for imagefile in ground.keys():
        img = cv2.imread(DB_DIR + '/' + DB + '/' + imagefile)

        num_gt = process_image(img, kp_detector, kp_descriptor, \
            METRIC, TH, MIN_CLUSTER_PTS, False)

        if num_gt >=1:
            if ground[imagefile] == '1':
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if ground[imagefile] == '1':
                FN = FN + 1
            else:
                TN = TN + 1


    # Compute performance
    TPR = float(TP)/(TP+FN) # True positive rate
    FPR = float(FP)/(FP+TN) # False positive rate

    return TPR, FPR

def all_experiments():
    ''' Run all the possible combinations of
        Keypoint Detectors and Keypoint Descriptors in a dataset '''

    # List of possible Keypoint Detectors and Keypoint Descriptors
    kp_detectors = ['SIFT','SURF','STAR','ORB']
    kp_descriptors = ['SIFT','SURF','ORB','BRIEF','BRISK','FREAK']

    # Try all type of configurations
    for detector in kp_detectors:
        for descriptor in kp_descriptors:
            try:
                print('Copy-Move Forgery Detection performance:')
                print('Detector: %s' % detector)
                print('Descriptor: %s' % descriptor)
                t0 = time.time()
                TPR, FPR = generic_experiment(detector,descriptor)
                t1 = time.time()
                print('Computational time: %f s' % (t1-t0))
                print('TPR = %f\nFPR = %f' % (TPR*100,FPR*100))
            except:
                print('%s,%s error' % (detector, descriptor))

if __name__ == '__main__':
    all_experiments()
