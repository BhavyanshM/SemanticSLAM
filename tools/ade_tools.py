import os
import json
import numpy as np
import cv2
import csv
import random
import torch

import yaml

from shutil import copyfile
# from utils import utils_ade20k
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Function used to determine if an image has an object from a pre-defined "required classes" list. If the image
# has an instance of a required class then it is included in the subset of images.
def full_mask_subset():
    jsons_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/jsons/'
    from_train_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/train/'
    to_train_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade_door_subset/train/'
    from_val_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/val/'
    to_val_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade_door_subset/val/'
    from_train_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/train_labels/'
    to_train_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade_door_subset/train_labels/'
    from_val_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/val_labels/'
    to_val_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade_door_subset/val_labels/'
    required_classes = ['door', 'handle', 'knob', 'hinge', 'door frame']

    # running helper function on train subset
    check_for_req_class_and_copy_image(from_train_dir, from_train_labels_dir, to_train_dir, to_train_labels_dir,
                                       jsons_dir, required_classes)

    # running helper function on val subset
    check_for_req_class_and_copy_image(from_val_dir, from_val_labels_dir, to_val_dir, to_val_labels_dir,
                                       jsons_dir, required_classes)

# Helper function for full_mask_subset()
def check_for_req_class_and_copy_image(from_img_dir, from_msk_dir, to_img_dir, to_msk_dir, jsons_dir, required_classes):
    jsons_list = os.listdir(jsons_dir)

    for filename in os.listdir(from_img_dir):
        if filename[:-4] + '.json' in jsons_list:
            with open(jsons_dir + filename[:-4] + '.json', 'r') as f:
                try:
                    input_info = json.load(f)
                    # print(input_info)
                    contents = input_info['annotation']['object']
                    names = [x['name'] for x in contents]
                    req_classes_mask = np.isin(names, required_classes)
                    if req_classes_mask.sum() > 0:
                        copyfile(from_img_dir + filename, to_img_dir + filename) # saving original image
                        copyfile(from_msk_dir + filename[:-4] + '_L.png', to_msk_dir + filename[:-4] + '_L.png')
                except:
                    print('failed on ', filename)

# Quick way to rename files according to  hyperseg file naming convention (labels with a _L.png at the end)
def rename_files():
    train_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/ADE_all_object_classes_w_req_parts_grayscale/trainannot/'
    val_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/ADE_all_object_classes_w_req_parts_grayscale/valannot/'
    for file in os.listdir(train_dir):
        os.rename(train_dir + file, train_dir + file[:-6] + '.png')
    for file in os.listdir(val_dir):
        os.rename(val_dir + file, val_dir + file[:-6] + '.png')

# Quick way to convert images from jpg to png.
# PNG is a nice format for training images since PNG images undergo lossless compression
def jpg_to_png():
    train_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/train/'
    new_train_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/train_png/'

    trainannot_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/ADE/trainannot/'
    new_trainannot_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/resized_ADE/trainannot/'

    val_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/val/'
    valannot_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/ADE/valannot/'

    new_val_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_ade/val_png/'
    new_valannot_dir = '/home/masselmeier/Desktop/semantic_models/light-weight-refinenet/data/resized_ADE/valannot/'

    for file in os.listdir(train_dir):
        img = cv2.imread(train_dir + file)
        cv2.imwrite(new_train_dir + file[:-4] + '.png', img)
    for file in os.listdir(val_dir):
        img = cv2.imread(val_dir + file)
        cv2.imwrite(new_val_dir + file[:-4] + '.png', img)

    for file in os.listdir(trainannot_dir):
        img = cv2.imread(trainannot_dir + file, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(new_trainannot_dir + file[:-4] + '.png', img)
    for file in os.listdir(valannot_dir):
        img = cv2.imread(valannot_dir + file, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(new_valannot_dir + file[:-4] + '.png', img)

# Function to extract all of the .json files throughout the ADE dataset master folder and put them all in one folder
# for easy searching or parsing
def extract_jsons():
    ade_train_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/images/ADE/training/'
    ade_val_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/images/ADE/validation/'
    json_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/jsons/'

    extract_split_jsons(ade_train_dir, json_dir)
    extract_split_jsons(ade_val_dir, json_dir)

# helper function for extract_jsons()
def extract_split_jsons(ade_split_dir, json_dir):
    for group in os.listdir(ade_split_dir):
        group_dir = ade_split_dir + group
        for directory_name in os.listdir(group_dir):
            category_dir = group_dir + '/' + directory_name
            for filename in os.listdir(category_dir):
                print('filename: ', category_dir + '/' + filename)
                if filename.endswith('.json'):
                    copyfile(category_dir + '/' + filename, json_dir + '/' + filename)


# Verified way of reading in the provided ADE masks. Code partially taken from repository of
# the authors of the ADE dataset (https://github.com/CSAILVision/ADE20K/blob/main/notebooks/ade20k_starter.ipynb)
def ADE_mask_modifier():
    # I did not know what the correct index was for these so I manually found them in a yaml
    req_classes = {'wall': -1, 'floor, flooring': -1, 'ceiling': -1, 'windowpane, window': -1, 'cabinet': -1,
                   'person, individual, someone, somebody, mortal, soul': -1, 'door': -1, 'table': -1,
                   'curtain, drape, drapery, mantle, pall': -1, 'chair': -1, 'stairs, steps': -1,
                   'stairway, staircase': -1}
    with open(r'/home/masselmeier/Desktop/Datasets/yamls/ADE_entire_class_list.yaml') as file:
        ADE_class_dict = yaml.load(file, Loader=yaml.FullLoader)
    for key in ADE_class_dict.keys():
        if key in req_classes.keys():
            req_classes[key] = ADE_class_dict[key]
    req_indices = req_classes.values()
    print('req_classes: ', req_classes)

    ade_train_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/images/ADE/training/'
    img_train_dir = '/home/masselmeier/Desktop/Datasets/ADE_chao_seg_masked_ICP/images/train/'
    mask_train_dir = '/home/masselmeier/Desktop/Datasets/ADE_chao_seg_masked_ICP/masks/train/'
    ade_val_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/images/ADE/validation/'
    img_val_dir = '/home/masselmeier/Desktop/Datasets/ADE_chao_seg_masked_ICP/images/val/'
    mask_val_dir = '/home/masselmeier/Desktop/Datasets/ADE_chao_seg_masked_ICP/masks/val/'

    masked_pxl_cutoff_ratio = 0.1

    make_custom_ADE_masks(ade_train_dir, img_train_dir, mask_train_dir, req_indices, req_classes, masked_pxl_cutoff_ratio)
    make_custom_ADE_masks(ade_val_dir, img_val_dir, mask_val_dir, req_indices, req_classes, masked_pxl_cutoff_ratio)

def make_custom_ADE_masks(ade_split_dir, img_split_dir, msk_split_dir, req_indices, req_classes, masked_pxl_cutoff_ratio):
    double_door_index = 783

    for group in os.listdir(ade_split_dir):
        group_dir = ade_split_dir + group
        for directory_name in os.listdir(group_dir):
            category_dir = group_dir + '/' + directory_name
            for filename in os.listdir(category_dir):
                if filename.endswith('.json'):
                    fileseg = filename.replace('.json', '_seg.png')
                    with Image.open(category_dir + '/' + fileseg) as io:
                        seg = np.array(io)
                    # Obtain the segmentation mask, built from the RGB channels of the _seg file
                    R = seg[:, :, 0]
                    G = seg[:, :, 1]
                    B = seg[:, :, 2]
                    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))
                    print('mask before: ', ObjectClassMasks)
                    ObjectClassMasks_shape = np.shape(ObjectClassMasks)
                    ObjectClassMasks = torch.from_numpy(ObjectClassMasks)
                    ObjectClassMasks[ObjectClassMasks == double_door_index] = req_classes['door'] # turning double door annotations into door annotations
                    print('mask after door switch: ', ObjectClassMasks)

                    deleted_class_mask = ~torch.any(
                        torch.stack([torch.eq(ObjectClassMasks, aelem).logical_or_(torch.eq(ObjectClassMasks, aelem)) for
                             aelem in req_indices], dim=0),
                        dim=0)
                    ObjectClassMasks[deleted_class_mask] = 0
                    count = 1
                    for index in req_indices:
                        ObjectClassMasks[ObjectClassMasks == index] = count
                        count += 1
                    print('mask after: ', ObjectClassMasks)
                    ratio = torch.count_nonzero(ObjectClassMasks) / (ObjectClassMasks_shape[0]*ObjectClassMasks_shape[1])
                    if ratio > masked_pxl_cutoff_ratio:
                        img_name = filename.replace('.json', '.jpg')
                        img = Image.open(category_dir + '/' + img_name)
                        img.save(img_split_dir + img_name.replace('.jpg', '.png'))
                        mask = Image.fromarray(np.uint8(ObjectClassMasks.numpy()))
                        mask.save(msk_split_dir + img_name.replace('.jpg', '.png'))
                    print('ratio: ', ratio)

def create_train_val_test_splits():
    image_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/indoor_subset/indoor_subset_images/'
    mask_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/indoor_subset/indoor_subset_masks/'

    train_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/train/'
    val_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/val/'
    test_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/test/'

    train_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/train_labels/'
    val_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/val_labels/'
    test_labels_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/test_labels/'

    images = os.listdir(image_dir)
    print(len(images))
    train_num = len(images) * .7
    val_num = len(images) * .2
    random.shuffle(images)
    for i in range(0, len(images)):
        image = images[i]
        if i < train_num:
            copyfile(image_dir + image, train_dir + image)
            copyfile(mask_dir + image.split('.')[0] + '_L.png', train_labels_dir + image.split('.')[0] + '_L.png')
        elif train_num <= i < train_num + val_num:
            copyfile(image_dir + image, val_dir + image)
            copyfile(mask_dir + image.split('.')[0] + '_L.png', val_labels_dir + image.split('.')[0] + '_L.png')
        else:
            copyfile(image_dir + image, test_dir + image)
            copyfile(mask_dir + image.split('.')[0] + '_L.png', test_labels_dir + image.split('.')[0] + '_L.png')

# helper function to find an image with a given instance
def instance_finder():
    desired_class = "double door"
    ade_train_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/images/ADE/training/'
    for group in os.listdir(ade_train_dir):
        group_dir = ade_train_dir + group
        for directory_name in os.listdir(group_dir):
            category_dir = group_dir + '/' + directory_name
            for filename in os.listdir(category_dir):
                try:
                    if filename.endswith('.json'):
                        with open(category_dir + '/' + filename, 'r') as f:
                            input_info = json.load(f)
                            contents = input_info['annotation']['object']
                            names = [x['raw_name'] for x in contents]
                            if desired_class in names:
                                print(desired_class + ' in: ', category_dir + '/' + filename)
                except:
                    print('failed on: ', category_dir + '/' + filename)

# instance class counter for a subset of ADE
def class_counter():
    with open(r'/home/masselmeier/Desktop/Datasets/indoor_classes/classes_with_colors.yaml') as file:
        class_color_dict = yaml.load(file, Loader=yaml.FullLoader)
        for key in class_color_dict.keys():
            color_string = class_color_dict[key]
            # print('string: ', color_string)
            # print('type:', type(color_string))
            # color_array = color_string.split(", ")
            # print('array: ', color_array)
            class_color_dict[key] = eval(color_string)
    class_occurrence_dict = {}
    for key in class_color_dict.keys():
        class_occurrence_dict[key] = 0
    class_image_dict = {}
    for key in class_color_dict.keys():
        class_image_dict[key] = 0
    class_image_added_dict = {}
    class_area_dict = {}
    for key in class_color_dict.keys():
        class_area_dict[key] = 0
    total_area = 0
    for key in class_color_dict.keys():
        class_image_added_dict[key] = 0

    #for directoryname in os.listdir(group_dir):
    directoryname = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/indoor_ADE/val/'
    json_dir = '/home/masselmeier/Desktop/Datasets/ADE20K_2021_17_01/jsons'

    for filename in os.listdir(directoryname):
        # print('filename: ', category_dir + '/' + filename)
        json_name = filename.split('.')[0] + '.json'
        #if filename.endswith('.json'):
        try:
            print('looking for: ', json_dir + '/' + json_name)
            with open(json_dir + '/' + json_name, 'r') as f:
                print('found')
                input_info = json.load(f)
                contents = input_info['annotation']['object']
                names = [x['name'] for x in contents]
                for name in names:
                    if name in class_occurrence_dict:
                        class_occurrence_dict[name] = class_occurrence_dict[name] + 1
                        class_image_dict[name] = class_image_dict[name] + 1
        except:
            print('not found')
    print('class occurrences: ')
    for key in class_occurrence_dict:
        print(key, ": ", class_occurrence_dict[key])
        print(key, ": ", class_image_dict[key])
        print("")


def split_file_name(filename):
    dot_split_length = len(filename.split('.'))
    split_string = filename.split('.', dot_split_length - 1)
    # print('split string: ', split_string)
    if dot_split_length == 3:
        image_name = '.'.join(split_string[:-1])
    else:
        image_name = split_string[0]
    return image_name

def create_train_val_test_text_files():
    TRAIN_DIR = '/home/masselmeier/Desktop/Datasets/ADE_chao_seg_masked_ICP/'

    #TRAIN_DIR = '/home/masselmeier/Desktop/semantic_models/models/research/deeplab/datasets/ADE_spec_classes/'
    '''
    # training
    # trainval_file = open(TRAIN_DIR + 'trainval.txt', 'a+')
    train_file = open(TRAIN_DIR + 'index/train.txt', 'a+')
    val_file = open(TRAIN_DIR + 'index/val.txt', 'a+')
    for filename in os.listdir(TRAIN_DIR + 'train'):
        train_file.write(filename.split('.')[0] + "\n")
        # trainval_file.write(filename.split('.')[0] + "\n")
    train_file.close()
    for filename in os.listdir(TRAIN_DIR + 'val'):
        val_file.write(filename.split('.')[0] + "\n")
        #trainval_file.write(filename.split('.')[0] + "\n")
    val_file.close()
    # trainval_file.close()
    '''
    train_file = open(TRAIN_DIR + 'train.txt', 'a+')
    val_file = open(TRAIN_DIR + 'val.txt', 'a+')
    for filename in os.listdir(TRAIN_DIR + 'images/train'):
        train_file.write('train/' + filename + ',' + 'trainannot/' + filename + "\n")
    train_file.close()
    for filename in os.listdir(TRAIN_DIR + 'images/val'):
         val_file.write('val/' + filename + ',' + 'valannot/' + filename + "\n")
    val_file.close()

def rename_weird_file():
    img_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_chao/train/'
    mask_dir = '/home/masselmeier/Desktop/semantic_models/hyperseg/data/full_chao/train_labels/'
    for img in os.listdir(img_dir):
        dot_split_length = len(img.split('.'))
        if dot_split_length == 3:
            os.rename(img_dir + img, img_dir + img.replace('.', '_', 1))
    for img in os.listdir(mask_dir):
        dot_split_length = len(img.split('.'))
        if dot_split_length == 3:
            os.rename(mask_dir + img, mask_dir + img.replace('.', '_', 1))

create_train_val_test_text_files()