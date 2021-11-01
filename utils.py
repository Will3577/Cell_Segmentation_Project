import cv2
import numpy as np
from skimage.morphology import reconstruction
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy import ndimage
import matplotlib.pyplot as plt 
import os
import imageio
import gdal

def mk_dirs(path):
  if not os.path.isdir(path):
      os.makedirs(path)

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

def extract_boundary(mask:np.ndarray,show_result:bool=False) -> np.ndarray:
    '''
    Extract the outer cell boundary from imput mask

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

# img_folder = '/content/Sequences_p/pseudo_masks/01/'
# des = '/content/test.gif'
def to_gif(img_folder:str, des:str):
    '''
    Warning: This function may only work on Colab
    
    Args:
        img_folder: folder that contain images to be compressed to gif 
        des: destination for .gif file
    Output:
        saved .gif file on destination
    '''
    filenames = sorted(os.listdir(img_folder))
    # print(filenames)
    images = []

    for filename in filenames:
        if filename.split('.')[-1]=='tif':
          images.append(gdal.Open(img_folder+filename).ReadAsArray())
        else:
          images.append(imageio.imread(img_folder+filename))
    imageio.mimsave(des, images)

# get all pos on image for given label 
def get_pos_list(img:np.array, label:int) -> [tuple]:
    '''
    Args: 
    img: TRA image(.tif) or instance segmentation image in np.array type
    label: label for a unique cell
    '''
    return list(zip(*np.where(img==label)))

# calculate the centroid of a list of positions
def get_centroid(pos:[tuple],dtype:str='float') -> tuple:
    x, y = zip(*pos)
    l = len(x)
    if dtype=='int':
        return round(sum(x)/l), round(sum(y)/l)
    else:
        return sum(x)/l, sum(y)/l

# get all centroids in the given image
def get_all_centroids(img:np.array) -> {tuple}:
    '''
    Args:
    img: image with unique number represent unique cell
    Output:
    dictionary with key:label,
                    value:(total pixels for this label, corresponding centroid)
    '''
    labels = np.unique(img)
    output = {}
    for label in labels:
        pos_list = get_pos_list(img,label)
        n_pixels = len(pos_list)
        # filter the background label
        if n_pixels<100000:
            centroid = get_centroid(pos_list,'int')
            output[label] = (n_pixels,centroid)
    return output





















# In case you want to test functions, 
# run !python utils.py and modify bellow code to get result
if __name__ == '__main__':
    print("print any function you want")