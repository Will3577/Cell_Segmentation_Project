# Sample usage can be viewed in 9517_mitosis_gen.ipynb

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_two_imgs(im1:np.array, im2:np.array, im1_title:str="im1", im2_title:str="im2"):
    f, axarr = plt.subplots(1,2,figsize=(15,15))
    axarr[0].imshow(im1)
    axarr[0].set_title(im1_title)
    axarr[1].imshow(im2)
    axarr[1].set_title(im2_title)
    plt.show()

def get_consecutive_imgs(im_number:int,path:str,prefix:str='.png') -> (np.array,np.array):
    if prefix=='.tif':
        current = cv2.imread(path+"{0:0=3d}".format(im_number)+prefix,-1)
        next = cv2.imread(path+"{0:0=3d}".format(im_number+1)+prefix,-1)
    else:
        current = cv2.imread(path+"{0:0=3d}".format(im_number)+prefix)
        next = cv2.imread(path+"{0:0=3d}".format(im_number+1)+prefix)
    return current, next

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
        # print(pos_list)
        n_pixels = len(pos_list)
        # filter the background label
        if n_pixels<100000:
            centroid = get_centroid(pos_list,'int')
            output[label] = (n_pixels,centroid)
    return output

# ------------------------------For ML mitosis detiction-------------------------------------
from sklearn.cluster import KMeans
import math
from sklearn.cluster import DBSCAN

# calculate euclidean distance between two position
def euc_dist(a:tuple,b:tuple) -> float:
    return np.linalg.norm(np.array(a)-np.array(b))

# Code from https://cs.stackexchange.com/questions/85929/efficient-point-grouping-algorithm/86040
def cluster(data, epsilon, N): #DBSCAN, euclidean distance
    db     = DBSCAN(eps=epsilon, min_samples=N).fit(data)
    labels = db.labels_ #labels of the found clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
    clusters   = [data[labels == i] for i in range(n_clusters)] #list of clusters
    return clusters, n_clusters

def get_pos(centroid_dict:{tuple}) -> [tuple]:
    X = []
    for tup in centroid_dict.values():
        X.append(tup[1])
    return X

# use child positions to inference the posision of parent
def group_child(centroid_dict:{tuple}) -> [tuple]:
    n_child = len(centroid_dict.keys())
    # print("t")
    k = math.ceil(n_child/2)
    # print("t")
    X = get_pos(centroid_dict)
    if len(X)==0:
        return np.array([])
    # print("t")
    X = np.array(X)
    # print(X)
    # TODO possible to generate incorrect clustering
    # ----------------DBSCAN clustering-------------------------
    clusters, n_clusters = cluster(X,40,2)
    # print(n_clusters,np.array(clusters).shape, np.array(clusters))
    # centers = np.mean(np.array(clusters),axis=1)
    centers = []
    for cl in clusters:
        pos = [int(np.mean(cl[:,0])),int(np.mean(cl[:,1]))]
        centers.append(pos)
        if len(cl)==1:
            print("Only 1 child is grouped, pos: ",pos)
        elif len(cl)>2:
            print("Warning! more than two children are grouped, pos: ",pos)
        
    # ----------------------end----------------------------------
    # -------------kmeans clustering and filtering---------------------
    # kmeans = KMeans(n_clusters=k, random_state=0,algorithm='full').fit(X)
    # centers = kmeans.cluster_centers_.tolist()

    # # filter child
    # labels = kmeans.labels_
    # print(len(labels),kmeans.get_params)
    # unique, counts = np.unique(labels, return_counts=True)
    # count_dict = dict(zip(unique, counts))
    # for id, count in enumerate(count_dict):
    #     if count==1:
    #         # del centers[id]
    #         print("1",centers[id])
    #     elif count>2:
    #         print("Warning! more than two children is grouped ",centers[id],)
    # ---------------------end kmeans---------------------------------
    # print(centers)
    return np.array(centers)

# filter the parents so that only true parents are preserved
def filter_parent(parent_dict:{tuple}, pseudo_parents:[tuple]) -> [tuple]:
    parent_centroids = get_pos(parent_dict)
    # pseudo_parents = group_child(child_dict)
    dist_dict = {}
    # Take effect when number of parent is larger than the number of pseudo parent
    if len(parent_centroids)>=len(pseudo_parents):
        for pseudo_pos in pseudo_parents:
            min_dist = math.inf
            for parent_pos in parent_centroids:
                dist = euc_dist(pseudo_pos,parent_pos)
                if min_dist>dist:
                    min_dist = dist
                    dist_dict[str(pseudo_pos)] = (dist, parent_pos)
        output = []
        for dist, p_pos in dist_dict.values():
            output.append(p_pos)
        return output
    else:
        print("Warning! n_parent < n_pseudo_parent")
        return parent_centroids


from PIL import Image
# crop the image into small pieces by centroids
def crop_by_centroid(img:np.array, centroid:tuple, crop_size:tuple=(40,40)) -> np.array:
    pil_img = Image.fromarray(np.uint8(img))
    width, height = crop_size
    c_x, c_y = centroid
    x = round(c_x-width//2)
    y = round(c_y-height//2)
    area = (x,y,x+width,y+height)
    cropped_img = pil_img.crop(area)
    # cropped_img.show()
    return cropped_img
# ---------------------------------end---------------------------------------------

import os
from tqdm import tqdm
def mk_dirs(path):
  if not os.path.isdir(path):
      os.makedirs(path)

def generate_mitosis_imgs(save_dir:str,crop_size:tuple=(40,40),f_name:int=2):
    total_imgs = 0
    mk_dirs(save_dir)
    for im_name in tqdm(range(91)):
        # img_folder = save_dir+"t{0:0=3d}".format(im_name)+'/'
        img_folder = save_dir+'0'+str(f_name)+'/'
        mk_dirs(img_folder)
        curr_dir = img_folder+'curr/'
        next_dir = img_folder+'next/'
        mk_dirs(curr_dir)
        mk_dirs(next_dir)

        curr_tra, next_tra = get_consecutive_imgs(im_name,'/content/COMP9517_Project/Sequences/0'+str(f_name)+'_GT/TRA/man_track','.tif')
        curr_img, next_img = get_consecutive_imgs(im_name,'/content/COMP9517_Project/Sequences_p/Sequences_Input/0'+str(f_name)+'_stretch_colorized/t')
        
        diff = set(next_tra.flatten())-set(curr_tra.flatten())
        rev_diff = set(curr_tra.flatten())-set(next_tra.flatten())

        masked_next = np.zeros(next_tra.shape)
        for id in diff:
            masked_next[next_tra==id] = id

        masked_tra = np.zeros(curr_tra.shape)
        for id in rev_diff:
            masked_tra[curr_tra==id] = id

        child_dict = get_all_centroids(masked_next)
        pseudo_parents_list = group_child(child_dict)
        parent_dict = get_all_centroids(masked_tra)
        mitosis_pos_list = filter_parent(parent_dict,pseudo_parents_list)

        total_imgs+=len(mitosis_pos_list)
        if len(mitosis_pos_list)!=len(pseudo_parents_list):
            print("not equal!")
            for pos in mitosis_pos_list: #TODO potential improvement by using both mitosis list and pseudo parent list
                cropped_curr = crop_by_centroid(curr_img,pos,crop_size)
                cropped_next = crop_by_centroid(next_img,pos,crop_size)
                cropped_curr.save(curr_dir+str(pos)+".jpg")
                cropped_next.save(next_dir+str(pos)+".jpg")
        else:
            for idx in range(len(mitosis_pos_list)): #TODO potential improvement by using both mitosis list and pseudo parent list
                pos = mitosis_pos_list[idx]
                pos_next = pseudo_parents_list[idx]
                cropped_curr = crop_by_centroid(curr_img,pos,crop_size)
                cropped_next = crop_by_centroid(next_img,pos,crop_size)
                cropped_curr.save(curr_dir+str(pos)+".jpg")
                cropped_next.save(next_dir+str(pos)+".jpg")
    print("total number of mitosis: ", total_imgs)


def generate_normal_imgs(save_dir:str,crop_size:tuple=(40,40),f_name:int=1):
    total_imgs = 0
    max_imgs = 206
    mk_dirs(save_dir)

    for im_name in range(80,81):
        # img_folder = save_dir+"t{0:0=3d}".format(im_name)+'/'
        img_folder = save_dir+'0'+str(f_name)+'/'
        mk_dirs(img_folder)
        curr_dir = img_folder+'curr/'
        next_dir = img_folder+'next/'
        mk_dirs(curr_dir)
        mk_dirs(next_dir)

        curr_tra, next_tra = get_consecutive_imgs(im_name,'/content/COMP9517_Project/Sequences/0'+str(f_name)+'_GT/TRA/man_track','.tif')
        curr_img, next_img = get_consecutive_imgs(im_name,'/content/COMP9517_Project/Sequences_p/Sequences_Input/0'+str(f_name)+'_stretch_colorized/t')
        
        diff = set(next_tra.flatten())-set(curr_tra.flatten())
        rev_diff = set(curr_tra.flatten())-set(next_tra.flatten())

        masked_next = next_tra.copy()
        for id in diff:
            masked_next[next_tra==id] = 0

        masked_tra = curr_tra.copy()
        for id in rev_diff:
            masked_tra[curr_tra==id] = 0

        # child_dict = get_all_centroids(masked_next)
        # pseudo_parents_list = group_child(child_dict)
        parent_dict = get_all_centroids(masked_tra)
        parent_list = get_pos(parent_dict)
        print(len(parent_list))
        # mitosis_pos_list = filter_parent(parent_dict,pseudo_parents_list)
        for idx in range(len(parent_list)):
            pos = parent_list[idx]
            cropped_curr = crop_by_centroid(curr_img,pos,crop_size)
            cropped_next = crop_by_centroid(next_img,pos,crop_size)
            cropped_curr.save(curr_dir+str(pos)+".jpg")
            cropped_next.save(next_dir+str(pos)+".jpg")
            total_imgs+=1
        if total_imgs>max_imgs:
            break
        
    print("total number of cells on track: ", total_imgs)