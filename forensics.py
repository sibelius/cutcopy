import cv2
import numpy as np
from scipy.spatial.distance import pdist
import sys

if __name__ == '__main__':
    # Load image gray scale
    img = cv2.imread(sys.argv[1],0)

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # Detect Keypoints and Compute Descriptors
    kp1, des1 = sift.detectAndCompute(img, None)

    # Calculate matches between a keypoint and k=3 more close keypoints
    matches = bf.knnMatch(des1, des1, k=3)

    # The first match is invalid, because is going to be the same Keypoint

    # We apply the Lowe and Amerini method to select good matches
    ratio = 0.5
    mkp1, mkp2 = [], []
    for m in matches:
        if m[1].distance < m[2].distance * ratio:
            m = m[1]

            # Check if the keypoints are spatial separated
            if pdist( np.array([kp1[m.queryIdx].pt, kp1[m.trainIdx].pt]) ) > 10:
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp1[m.trainIdx] )

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)

    green = (0, 255, 0)
    r = 2
    thickness = 3
    col = green

    vis = img.copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    p1 = np.int32(p1)
    p2 = np.int32(p2)

    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
        cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
        cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
        cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

        cv2.line(vis, (x1, y1), (x2,y2), col)

    win = 'Image Forensics'
    cv2.imshow(win, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


