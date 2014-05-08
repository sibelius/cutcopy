'''

This file contains some tests about the matching of objects for cut-copy
detection

'''

import cv2
import numpy as np
import scipy.spatial

def normalize_descriptor(des1):
    ''' Apply the 2-norm to the descriptor '''

    norm = np.apply_along_axis(np.linalg.norm, 1, des1)
    return des1 / norm[:,np.newaxis]

def appendimages(img1, img2):
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]

    temp1 = img1.copy()
    temp2 = img2.copy()

    if rows1 < rows2:
        temp1.resize((rows2, img1.shape[1]))
    else:
        temp2.resize((rows1, img2.shape[1]))

    return np.hstack((temp1, temp2))

def print_dmatch(m):
    print('imgIdx\tqueryIdx\ttrainIdx\tdistance')
    print(m.imgIdx, m.queryIdx, m.trainIdx, m.distance)

# threshold used for g2NN test
dr2 = 0.6

# number of matches

kp1
# sift matching
des1 = normalize_descriptor(des1)
des2t = des1.T
p1 = []
p2 = []
num = 0
match = np.zeros((des1.shape[0]))
if len(des1) > 1:
    for i in range(des1.shape[0]):
        dotprods = des1[i,:].dot(des2t)
        dotprods = np.arccos(dotprods)
        vals = np.sort(dotprods)
        indx = np.argsort(dotprods)

        j = 1
        while(vals[j] < dr2 * vals[j+1]):
            j = j + 1
        if j > 1:
            print("%f %f %f" %(j,vals[j],vals[j+1])

        for k in range(1,j-1):
            print('match')
            match[i] = indx[k]
#            if scipy.spatial.distance.pdist( \
#                [ \
#                [kp1[i].pt[0],kp1[i].pt[1]], \
#                [kp1[match[i]].pt[0],kp1[match[i]].pt[1]] \
#                ]) > 10:
#                p1.append([[kp1[i].pt[1]],[kp1[i].pt[0]],[1]])
#                p2.append([[kp1[match[i]].pt[1]],[kp1[match[i]].pt[0]],[1]])
#                num = num + 1

distRatio = 0.6
des2t = des1.T
match = np.zeros((des1.shape[0]))
for i in range(des1.shape[0]):
    dotprods = des1[i,:].dot(des2t)
    vals = np.sort(np.arccos(dotprods))
    indx = np.argsort(np.arccos(dotprods))

    if vals[0] < distRatio * vals[1]:
        match[i] = indx[0]
    else:
        match[i] = 0

img_lines = img3.copy()
cols1 = img1.shape[1]
for i in range(des1.shape[0]):
    if match[i] > 0:
        cv2.line(img_lines, \
            (int(kp1[i].pt[1]),int(kp2[match[i]].pt[1]+cols1)), \
            (int(kp1[i].pt[0]),int(kp2[match[i]].pt[0])), (0,0,0))














dr2 = 0.5
des2t = des1.T
match = np.zeros((des1.shape[0]))

try:
    for i in range(des1.shape[0]):
        dotprods = des1[i,:].dot(des2t)
        # Fix for python
        dotprods[np.nonzero(dotprods > 1)[0]] = 1.0
        dotprods[np.nonzero(dotprods < -1)[0]] = -1.0

        vals = np.sort(np.arccos(dotprods))
        indx = np.argsort(np.arccos(dotprods))

        j = 1
        while vals[j] < dr2 * vals[j+1]:
            j = j + 1

        for k in range(1,j-1):
            match[i] = indx[k]
except Warning:
    print 'Warning'
    print(i)




