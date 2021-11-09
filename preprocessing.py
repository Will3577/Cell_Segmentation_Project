import numpy as np

def max_filter(I, N):
    
    # Assume N is no less than 3 and odd.
    N_center = int((N - 1) / 2)
    
    # First, it should create a new 
    # image, let us call it image A, with the same size (number of pixel rows and columns) as the 
    # input image, which we call I.
    A = I.copy()
    
    # Second, the algorithm should go through the pixels of I one by 
    # one, and for each pixel (x,y) it must find the maximum gray value in a neighbourhood 
    # centered around that pixel, and write that maximum gray value in the corresponding pixel 
    # location (x,y) in A.
    width = I.shape[0]
    length = I.shape[1]
    
    for x in range(0, width):
        for y in range(0, length):
            
            max_value = I[x, y]
            
            N_top = x - N_center
            N_bottom = x + N_center
            N_left = y - N_center
            N_right = y + N_center
            
            if N_top < 0: N_top = 0
            if N_bottom >= width: N_bottom = width - 1
            if N_left < 0: N_left = 0
            if N_right >= length: N_right = length - 1
            
            for i in range(N_top, N_bottom + 1):
                for j in range(N_left, N_right + 1):
                    if I[i, j] > max_value:
                        max_value = I[i, j]
            
            A[x, y] = max_value
            
    return A

def min_filter(A, N):

    # Assume N is no less than 3 and odd.
    N_center = int((N - 1) / 2)

    # First, it should create a new 
    # image, let us call it image A, with the same size (number of pixel rows and columns) as the 
    # input image, which we call I.
    B = A.copy()

    # Second, the algorithm should go through the pixels of I one by 
    # one, and for each pixel (x,y) it must find the maximum gray value in a neighbourhood 
    # centered around that pixel, and write that maximum gray value in the corresponding pixel 
    # location (x,y) in A.
    width = A.shape[0]
    length = A.shape[1]

    for x in range(0, width):
        for y in range(0, length):
            
            min_value = A[x, y]
            
            N_top = x - N_center
            N_bottom = x + N_center + 1
            N_left = y - N_center
            N_right = y + N_center + 1
            
            if N_top < 0: N_top = 0
            if N_bottom > width: N_bottom = width
            if N_left < 0: N_left = 0
            if N_right > length: N_right = length
            
            for i in range(N_top, N_bottom):
                for j in range(N_left, N_right):
                    if A[i, j] < min_value:
                        min_value = A[i, j]
            
            B[x, y] = min_value
            
        return B

def subtract_image(img1,img2,M):
    width = img1.shape[0]
    length = img1.shape[1]
        
        
    O = img1.copy()
    for x in range(width):
        for y in range(length):
            if M == 0:
                O[x,y] = np.uint8(int(img1[x,y]) - int(img2[x,y]) + 65535)
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
        
    return B, O
