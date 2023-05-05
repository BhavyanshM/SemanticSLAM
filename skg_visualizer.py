import cv2
import os
from KITTI_Classes import *
import numpy as np

import torch
import spacy

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

def sync_labels():
    count = 0
    for x in label_files:
        if os.path.exists(img_path + x[:-4] + '.jpg'):
            print('Found: ', x[:-4] + '.jpg')
        else:
            count += 1
            # os.rename(facade_img_path + x, archive_path + x)
            print('Not Found : ', x[:-4] + '.jpg', count)

def visualize(img_files, img_path, label_files, label_path):
    for file in img_files:
        img = cv2.imread(img_path + file)
        labels = open(label_path + file[:-4] + '.txt')

        for l in labels:
            det = list(map(float, l.split(' ')))
            x, y, w, h = int(det[1] * img.shape[1]), int(det[2] * img.shape[0]), int(det[3] * img.shape[1]), int(det[4] * img.shape[0])
            # print(x,y,w,h)
            print(classes[int(det[0])], int(det[0]), x, y, w, h)

            cv2.rectangle(img, (x - int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), (24, 244, 100), 3)

        # print(file)
        cv2.imshow("Image", img)
        code = cv2.waitKeyEx(0)
        if code == 113:
            exit()

def extend_graph(detections, graph_matrix):

    graph = dict()    

    for i in detections:
        for j in detections:
            if i != j:
                min_index = min(i, j)
                max_index = max(i, j)
                if (min_index, max_index) not in graph:
                    graph[(min_index, max_index)] = 1
                else:
                    graph[(min_index, max_index)] += 1

                graph_matrix[min_index][max_index] += 1
                graph_matrix[max_index][min_index] += 1

            else:
                graph_matrix[i][j] += 1

    return graph_matrix


def generate_skg_matrix(label_path, label_files, names, count):

    graph = []
    objects = []

    graph_matrix = np.zeros((len(names), len(names)))

    for file in tqdm(label_files[:count]):
        labels = open(label_path + file)

        det_index_list = [int(line.split(' ')[0]) for line in labels]

        # det_name_list = [names[int(line.split(' ')[0])] for line in labels]

        # print(det_index_list)

        graph_matrix = extend_graph(det_index_list, graph_matrix)

    return graph_matrix

def compute_bert_similarity(tokenizer, model, word1, word2):
    # Encode the two words with BERT
    word1_tokens = tokenizer.encode(word1, add_special_tokens=False)
    word2_tokens = tokenizer.encode(word2, add_special_tokens=False)
    inputs = tokenizer.encode_plus(word1, word2, return_tensors='pt', add_special_tokens=True)
    outputs = model(**inputs)
    word1_embedding = outputs.last_hidden_state[0][1].detach().numpy() # Extract embedding for the first token of word1
    word2_embedding = outputs.last_hidden_state[0][2].detach().numpy() # Extract embedding for the first token of word2

    # Compute cosine similarity between the two embeddings
    similarity = (cosine_similarity([word1_embedding], [word2_embedding]) + 1) / 2

    return similarity[0][0]

def compute_word2vec_similarity(nlp, word1, word2):

    # Compute word similarity using word2vec from spacy    
    word1 = nlp(word1)
    word2 = nlp(word2)
    similarity = word1.similarity(word2)

    return similarity


def generate_bert_matrix(classes):

    with torch.no_grad():
        model = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        graph_matrix = np.zeros((len(classes), len(classes)))
        for i in tqdm(range(len(classes))):
            for j in range(len(classes)):
                if i != j:
                    sim = compute_bert_similarity(tokenizer, model, classes[i], classes[j])
                    graph_matrix[i][j] = sim
                    graph_matrix[j][i] = sim
                    # print("Similarity: ", classes[i], classes[j], sim)
        return graph_matrix

def generate_word2vec_matrix(classes):

    nlp = spacy.load('en_core_web_lg')

    graph_matrix = np.zeros((len(classes), len(classes)))
    for i in tqdm(range(len(classes))):
        for j in range(len(classes)):
            if i != j:
                sim = compute_word2vec_similarity(nlp, classes[i], classes[j])
                graph_matrix[i][j] = sim
                graph_matrix[j][i] = sim
                # print("Similarity: ", classes[i], classes[j], sim)
    return graph_matrix

def plot_heatmap(graph_matrix, upper_limit, names):
    plt.figure(figsize=(20, 20))

    sns.set(font_scale=1.3)

    # Use Seaborn to plot the confusion matrix with color range for 0 - 10000 only
    sns.heatmap(graph_matrix, annot=False, fmt='g', cmap='Blues', vmin=0, vmax=upper_limit, xticklabels=names, yticklabels=names, 
                square=True)
    plt.show()

if __name__ == "__main__":
    img_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/images/train2017/'
    label_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/labels/train2017/'
    archive_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/archive/'
    facade_img_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/images/facade/'
    facade_label_path = '/home/quantum/Workspace/Storage/Other/Temp/dataset/YOLO_Data/data/labels/facade/'

    #-For Visualizing COCO dataset ------------------------------------------------
    img_files = os.listdir(img_path)
    label_files = os.listdir(label_path)

    # -------------------- For visualizing FacadeLabelMe dataset ---------------------------------------
    # facade_img_files = sorted(os.listdir(facade_img_path))
    # facade_label_files = sorted(os.listdir(facade_label_path))
    # img_files = facade_img_files
    # label_files = facade_label_files
    # img_path = facade_img_path
    # label_path = facade_label_path

    # names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    #         'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #         'teddy bear', 'hair drier', 'toothbrush']

    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
            'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'window']


    print('Total Classes: ', len(classes))


    count = 10000
    upper_limit = count / 200 

    # visualize(img_files, img_path, label_files, label_path)

    # graph_matrix = generate_skg_matrix(label_path, label_files, classes, count)

    # bert_matrix = generate_bert_matrix(classes)

    word2vec_matrix = generate_word2vec_matrix(classes)


    plot_heatmap(word2vec_matrix, 0.5, classes)