#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from vocabulary_tree import VocabularyTree
import math

parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

train_path = args["trainingSet"]
training_names = os.listdir(train_path)

tree_depth = 3
tree_k = 10

image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

fea_det = cv2.xfeatures2d.SIFT_create()

des_list = []

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))
    # 提取SIFT特征
    kpts, des = fea_det.detectAndCompute(im, None)
    
    des_list.append((image_path, des))

downsampling = 2

print("Stacking and downsampling descriptors for vocabulary tree building...")
descriptors = des_list[0][1][::downsampling,:]

for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

print ("Building Vocabulary Tree: depth={}, k={}, descriptors={}".format(
    tree_depth, tree_k, descriptors.shape[0]))

# 构建词汇树
voc_tree = VocabularyTree(depth=tree_depth, k=tree_k)
voc_tree.build(descriptors)

numWords = voc_tree.num_leaves
print("Vocabulary tree built: {} leaf nodes (visual words)".format(numWords))

tf_histograms = np.zeros((len(image_paths), numWords), "float32")

for i in range(len(image_paths)):
    descriptors = des_list[i][1]
    words = voc_tree.quantize(descriptors)
    if len(words) > 0:
        word_hist = np.bincount(words, minlength=numWords).astype("float32")
        tf_histograms[i] = word_hist
    print("Processed image %d of %d" % (i+1, len(image_paths)))

nbr_occurences = np.sum( (tf_histograms > 0) * 1, axis = 0)
print('occur',nbr_occurences.shape)

idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
print('idf',idf)

# 计算TF-IDF特征
tfidf_features = tf_histograms*idf
im_features = preprocessing.normalize(tfidf_features, norm='l2')

# 构建加权倒排索引
weighted_inverted_index = [[] for _ in range(numWords)]
for doc_id in range(len(image_paths)):
    normalized_vector = im_features[doc_id]
    nonzero_words = np.nonzero(normalized_vector)[0]
    for w in nonzero_words:
        weight = float(normalized_vector[w])
        if weight > 0:
            weighted_inverted_index[w].append((doc_id, weight))

# 保存特征和索引
joblib.dump((im_features, image_paths, idf, numWords, voc_tree, weighted_inverted_index), "tree-bag-of-words.pkl", compress=3)
print("Saved vocabulary tree and features to tree-bag-of-words.pkl")
