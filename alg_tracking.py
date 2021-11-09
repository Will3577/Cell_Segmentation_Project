import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import *


def track_cells_in_folder(input_path, output_path, is_thresh):
    file_num = 0

    # input_path = "Sequences_p/0"+ str(folder_num) + "_ml_pred/"
    # output_path = "Sequences_p/0"+ str(folder_num) + "_ml_pred_alg_tra/"

    os.makedirs(output_path, exist_ok=True)

    images = load_images(input_path, -1)
    length = len(images)
    pre_ws_labels = 0
    used_labels = []
    minimal = -1
    maximum = -1

    for img in images:
        if is_thresh:
            img = remove_small_dots(img)
            img = fill_small_holes(img)

        if file_num == 0:
            if not is_thresh:
                img = binarize_and_optimize_image(img, 90, 65535)
            pre_ws_labels = apply_watershed(img, 15)

        else:
            if not is_thresh:
                img = binarize_and_optimize_image(img, 90, 65535)
            ws_labels1 = apply_watershed(img, 15) 
            label_info_list, ws_labels1_tracked = track_cells(pre_ws_labels, ws_labels1, 0.2, used_labels)
            pre_ws_labels = ws_labels1_tracked

        minimal, maximum = find_extreme_value(pre_ws_labels)
        pre_ws_labels = pre_ws_labels.astype(np.uint16)


        filename = output_path + "t{0:0=3d}".format(file_num) + ".tif"
        plt.imshow(pre_ws_labels)
        plt.show()

        cv2.imwrite(filename, pre_ws_labels)
        file_num += 1
        print(str(file_num) + "/" + str(length) + " - completed, min: " + str(minimal) + " max: " + str(maximum))


def get_cell_count_list(pathname):
    images = load_images(pathname, -1)
    cell_count_list = []
    for image in images:
        cell_count = get_cell_count(image)
        cell_count_list.append(cell_count)
    return cell_count_list

def get_cell_count(ws_label):
    cell_count = len(np.unique(ws_label)) - 1
    return cell_count

def get_average_size_list(pathname):
    images = load_images(pathname, -1)
    average_size_list = []
    for image in images:
        average_size = get_average_size(image)
        average_size_list.append(average_size)
    return average_size_list

def get_average_size(ws_label):
    ws_label_no_border = remove_border_object(ws_label)
    total_cell_number = len(np.unique(ws_label_no_border)) - 1
    total_size = 0
    for x in range(ws_label_no_border.shape[0]):
        for y in range(ws_label_no_border.shape[1]):
            if ws_label_no_border[x, y] != 0:
                total_size += 1

    average_size = int(float(total_size) / float(total_cell_number))
    return average_size

# def calculate_average_size_and_total_cell_number(ws_label):
#     total_cell_number = len(np.unique(ws_label)) - 1
#     total_size = 0
#     for x in range(ws_label.shape[0]):
#         for y in range(ws_label.shape[1]):
#             if ws_label[x, y] != 0:
#                 total_size += 1

#     average_size = int(float(total_size) / float(total_cell_number))
#     return total_cell_number, average_size

# Track the cells between two images (16 bit)
# output image with tracked labels
def track_cells(ws_labels0, ws_labels1, coverage, used_labels):

    label_info_list = collect_overlap_label_info(ws_labels0, ws_labels1, coverage, used_labels)

    ws_labels1_tracked = ws_labels1.copy()

    for x in range(ws_labels1_tracked.shape[0]):
        for y in range(ws_labels1_tracked.shape[1]):
            if ws_labels1_tracked[x, y] != 0:
                label = ws_labels1_tracked[x, y]
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

def collect_overlap_label_info(ws_labels0, ws_labels1, coverage, used_labels):

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

                if cur_label not in used_labels:
                    used_labels.append(cur_label)

                # If this pixel have overlapping labels
                if ol_label != 0:
                    ol_label_list = cur_label_info["ol_label_list"]

                    # Check if overlap label is already in the list, 
                    # if so, do increment.
                    # Else, do appending.
                    ol_label_count_increment(ol_label_list, ol_label)

    find_the_most_overlapping_cell(label_info_list, coverage, used_labels)
    reassign_repeated_label(label_info_list, used_labels)

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


def find_the_most_overlapping_cell(label_info_list, coverage, used_labels):
    
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
            new_label = max(used_labels) + 1
            label["chosen_overlap_label"] = new_label
            used_labels.append(new_label)


def reassign_repeated_label(label_info_list, used_labels):

    visited_labels = []
    repeated_labels = []
    for label_info in label_info_list:
        if label_info["chosen_overlap_label"] not in visited_labels:
            visited_labels.append(label_info["chosen_overlap_label"])
        else:
            repeated_labels.append(label_info["chosen_overlap_label"])

    for repeated_label in repeated_labels:
        for label_info in label_info_list:
            if label_info["chosen_overlap_label"] == repeated_label:
                new_label = max(used_labels) + 1
                label_info["chosen_overlap_label"] = new_label
                used_labels.append(new_label)

