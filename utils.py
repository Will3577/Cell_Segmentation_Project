import cv2
import numpy as np
from skimage.morphology import reconstruction
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt 

def fill_small_holes(thresh):
    kernel = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    return thresh
    
def remove_small_dots(thresh):
    kernel = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    return thresh

def remove_border_object(img):
    
    border = np.zeros(img.shape, dtype=np.uint8)
    height, width = border.shape

    for i in range(width):
        border[0,i] = img[0,i]
        border[height-1,i] = img[height-1,i]

    for j in range(height):
        border[j,0] = img[j,0]
        border[j,width-1] = img[j,width-1]
    
    rec_border = reconstruction(border, img)
        
    img_no_border = img - np.uint8(rec_border)
    
    return img_no_border

# Image needs to be in cv2 grayscale.
def binarize_and_optimize_image(img):

    thresh = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)[1]
    thresh = fill_small_holes(thresh)
    thresh = remove_small_dots(thresh)
    thresh = remove_border_object(thresh)
    
    return thresh

from scipy import ndimage

def extract_boundary(mask,show_result=False):
    '''
    Args:
        mask: 2d numpy array, 255 for foreground, 0 for background
        show_result: verbose to print image
    Output:
        image with only contours, 
        pixel value of 255 indicates contour, 0 indicates background
    '''
    d_mask = ndimage.binary_dilation(mask)
    res = np.zeros(d_mask.shape)
    res[d_mask==True] = 255
    res = res.astype('int32')
    output = np.subtract(res,mask)
    if show_result:
        plt.imshow(output,cmap='gray')
        plt.show()
    return output

# In case you want to test functions, 
# run !python utils.py and modify bellow code to get result
if __name__ == '__main__':
    print("print any function you want to test!")