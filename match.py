bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck=True)

matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)
explore_match('find_obj', img1,img2,kp_pairs)#cv2 shows image

cv2.waitKey()
cv2.destroyAllWindows()

