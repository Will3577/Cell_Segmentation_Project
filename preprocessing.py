import numpy as np
import cv2

def max_filter(I, N):

    kernel = np.ones((N, N), dtype=np.uint16)
    I = cv2.dilate(I, kernel, iterations=1)

    return I

def min_filter(I, N):

    kernel = np.ones((N, N), dtype=np.uint16)
    I = cv2.erode(I, kernel, iterations=1)

    return I

def subtract_image(img1,img2,M):
    width = img1.shape[0]
    length = img1.shape[1]
        
        
    O = img1.copy()
    for x in range(width):
        for y in range(length):
            if M == 0:
                O[x,y] = np.uint16(int(img1[x,y]) - int(img2[x,y]) + 65535)
            elif M == 1:
                O[x,y] = img1[x,y] - img2[x,y]
            
    return O

def remove_background(I, N, M):
    O = I.copy()
    if M == 0:
        A = max_filter(I, N)
        B = min_filter(A, N)
        O = subtract_image(I, B, M)
    elif M == 1:
        A = min_filter(I, N)
        B = max_filter(A, N)
        O = subtract_image(I, B, M)
        
    return O
