import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import *

# Track the cells between two images (16 bit)
# output image with tracked labels
def track_cells(ws_labels0, ws_labels1, coverage):

    label_info_list = collect_overlap_label_info(ws_labels0, ws_labels1, coverage)

    ws_labels1_tracked = ws_labels1.copy()

    for x in range(ws_labels1_tracked.shape[0]):
        for y in range(ws_labels1_tracked.shape[1]):
            if ws_labels1_tracked[x, y] != 0:
                label = ws_labels1_tracked[x, y]
                # overlap_label = label_info_list[label-1]["chosen_overlap_label"]
                label_info = find_label_in_list(label, label_info_list)

                assert(label_info != -1)
                ws_labels1_tracked[x, y] = label_info["chosen_overlap_label"]

    return label_info_list, ws_labels1_tracked
# Apply watershed, input should be thresholded image.
def apply_watershed(thresh, min_distance):
    distance = ndi.distance_transform_edt(thresh)
    peak_array = peak_local_max(distance, footprint=np.ones((15,15)), min_distance=min_distance, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(peak_array.T)] = True
    markers, _ = ndi.label(mask)
    ws_labels = watershed(-distance, markers, mask=thresh)
    return ws_labels

def collect_overlap_label_info(ws_labels0, ws_labels1, coverage):

    label_info_list = []

    # Go through every pixel
    for x in range(ws_labels1.shape[0]):
        for y in range(ws_labels1.shape[1]):

            cur_label = ws_labels1[x,y]
            ol_label = ws_labels0[x,y]

            if cur_label != 0:
                cur_label_info = find_label_in_list(cur_label, label_info_list)

                if cur_label_info != -1:
                    cur_label_info["cell_size"] += 1
                else:
                    cur_label_info = {"label": cur_label,
                                      "cell_size": 1, 
                                      "ol_label_list": [],
                                      "chosen_overlap_label": -1}
                    label_info_list.append(cur_label_info)

            # If this pixel have overlapping labels
                if ol_label != 0:
                    ol_label_list = cur_label_info["ol_label_list"]

                    # Check if overlap label is already in the list, 
                    # if so, do increment.
                    # Else, do appending.
                    ol_label_count_increment(ol_label_list, ol_label)

    find_the_most_overlapping_cell(label_info_list, coverage)

    return label_info_list

def find_label_in_list(label, label_info_list):

    for k in label_info_list:
        if k["label"] == label:
            return k

    return -1

def ol_label_count_increment(ol_label_list, ol_label):

    ol_label_found = False
    ol_label_index = 0

    k = 0
    for overlapping_label in ol_label_list:
        if overlapping_label["label"] == ol_label:
            ol_label_found = True
            ol_label_index = k
        k+=1

    if ol_label_found:
        ol_label_list[ol_label_index]["count"] += 1

    else:
        new_overlap_label = {"label": ol_label,
                            "count": 1}
        ol_label_list.append(new_overlap_label)


def find_the_most_overlapping_cell(label_info_list, coverage):
    
    unused_label = -1
    for label in label_info_list:
        if (unused_label < label["label"]):
            unused_label = label["label"]
    
    unused_label += 1

    for label in label_info_list:
        max_count = -1
        max_label = -1
        for overlap_label in label["ol_label_list"]:
            if overlap_label["count"] > max_count:
                max_count = overlap_label["count"]
                max_label = overlap_label["label"]

        calculated_coverage = -1
        if max_count != -1 :
            calculated_coverage = float(max_count) / float(label["cell_size"]) 
            
        if calculated_coverage >= coverage:
            label["chosen_overlap_label"] = max_label

        else:
            label["chosen_overlap_label"] = unused_label
            unused_label += 1



