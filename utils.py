from preprocessing import *
import cv2
import numpy as np
from skimage.morphology import reconstruction
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy import ndimage
import matplotlib.pyplot as plt 
import os
import math
import gdal
import imageio
# import gdal

def mk_dirs(path):
  if not os.path.isdir(path):
      os.makedirs(path)

# Load a image list from a folder (specify the folder_num, and flag for cv2.imread)
# 1 <= folder_num <= 5
def load_images(pathname, flag):
    img_list = []
    filenames = sorted(os.listdir(pathname))
    for file in filenames:
        img = cv2.imread((os.path.join(pathname, file)), flag)
        img_list.append(img)
    return img_list

def export_images(pathname, images):
    file_num = 0
    os.makedirs(pathname, exist_ok=True)
    for image in images:
        filename = pathname + "t{0:0=3d}".format(file_num) + ".png"
        # cv2.imwrite(filename, image)
        plt.imsave(filename, image)
        file_num += 1

def fill_small_holes(thresh):
    kernel = np.ones((3, 3), dtype=np.uint16)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    return thresh

def remove_small_dots(thresh):
    kernel = np.ones((3, 3), dtype=np.uint16)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    return thresh

def remove_border_object(img):
    
    border = np.zeros(img.shape, dtype=np.uint16)
    height, width = border.shape

    for i in range(width):
        border[0,i] = img[0,i]
        border[height-1,i] = img[height-1,i]

    for j in range(height):
        border[j,0] = img[j,0]
        border[j,width-1] = img[j,width-1]
    
    rec_border = reconstruction(border, img)
        
    img_no_border = img - np.uint16(rec_border)
    
    return img_no_border

def binarize_and_optimize_image(img, relative_threshold_low, threshold_high, gaussian_blur=True, r_small_dots=True, f_small_holes=True, r_border=False):

    
    if gaussian_blur:
        img = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)

    kernel = np.ones((7, 7), dtype=np.uint16)
    img = cv2.erode(img, kernel, iterations=1)

    low, high = find_extreme_value(img)
    thresh = cv2.threshold(img, low + relative_threshold_low, threshold_high, cv2.THRESH_BINARY)[1]
    if r_small_dots:
        thresh = remove_small_dots(thresh)
    if f_small_holes:
        thresh = fill_small_holes(thresh)
    if r_border:
        thresh = remove_border_object(thresh)
    
    return thresh

def contrast_stretching(img):
    c, d = find_extreme_value(img)
    
    a = 0
    b = 65535
    
    new_img = img.copy()
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            new_img[x,y] = (img[x,y] - c) * ((b - a)/(d - c)) + a

    return new_img

def find_extreme_value(img):

    c = img[0,0]
    d = img[0,0]


    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
                if img[x,y] < c:
                    c = img[x,y]
                if img[x,y] > d:
                    d = img[x,y]    
    return c, d

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

def plot_two_imgs(im1:np.array, im2:np.array, im1_title:str="im1", im2_title:str="im2"):
    f, axarr = plt.subplots(1,2,figsize=(15,15))
    axarr[0].imshow(im1)
    axarr[0].set_title(im1_title)
    axarr[1].imshow(im2)
    axarr[1].set_title(im2_title)
    plt.show()

# get all pos on image for given label 
def get_pos_list(img:np.array, label:int) -> [tuple]:
    '''
    Args: 
    img: TRA image(.tif) or instance segmentation image in np.array type
    label: label for a unique cell
    '''
    out = list(zip(*np.where(img==label)))
    # reverse the output list so that x=tuple[0], y=tuple[1]
    reversed = [t[::-1] for t in out]
    return reversed

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

from sklearn.cluster import KMeans
import math
# use child positions to inference the posision of parent
def group_child(centroid_dict:{tuple}) -> [tuple]:
    n_child = len(centroid_dict.keys())
    k = math.ceil(n_child/2)
    print(n_child/2,k)
    X = []
    for tup in centroid_dict.values():
        X.append(tup[1])
    X = np.array(X)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    return centers

def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)

def get_average(data:[int]) -> int:
    sum = 0
    for d in data:
      sum = sum + d
    ave = sum / len(data)
    return ave

def distance(pos1:tuple, pos2:tuple) -> int:
    square = math.pow(abs(pos1[0] - pos2[0]),2) + math.pow(abs(pos2[1] - pos2[1]),2)
    d = math.sqrt(square)
    return d

def save_txt(data:[float],Codename:int):
    tf = open("Sequences_p/displacement/"+str(Codename)+'_dis.txt','a')
    for d in data:
        tf.write(str(d)+'\n')
    tf.close()












# In case you want to test functions, 
# run !python utils.py and modify bellow code to get result
if __name__ == '__main__':
    print("print any function you want")
