import cv2
import numpy as np

if __name__ == '__main__':
    filename = 'ledemissiles1.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    WINDOW_NAME = 'win'


