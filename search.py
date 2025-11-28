#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

import argparse as ap
import cv2
import numpy as np
import os
import math
from sklearn.externals import joblib
from sklearn import preprocessing
from vocabulary_tree import VocabularyTree
from Inverted_index import compute_score_with_inverted_index
from RANSAC import ransac_verification, display_ransac_results
from Relevance_Feedback import collect_feedback_from_ids, apply_feedback

parser = ap.ArgumentParser(description="图像检索系统 - 基于词汇树和倒排索引")
parser.add_argument("-i", "--image", help="Path to query image", required="True")
parser.add_argument("-f", "--feedback", action="store_true", help="Enable relevance feedback")
parser.add_argument("--top-k", type=int, default=20, help="Number of results to display for feedback")
args = vars(parser.parse_args())
image_path = args["image"]
enable_feedback = args["feedback"]
top_k = args["top_k"]

im_features, image_paths, idf, numWords, voc_tree, inverted_index = joblib.load("tree-bag-of-words.pkl")

fea_det = cv2.xfeatures2d.SIFT_create()

orig_query = cv2.imread(image_path)
im = orig_query.copy()

# 提取查询图像的SIFT特征
kpts, des = fea_det.detectAndCompute(im, None)

words = voc_tree.quantize(des)
query_word_indices = np.array(words, dtype=np.int32)

test_features = np.zeros((1, numWords), "float32")
for w in query_word_indices:
    test_features[0][w] += 1

# 计算查询向量的TF-IDF特征
test_features = test_features * idf
test_features = preprocessing.normalize(test_features, norm='l2')

original_query_vector = test_features.copy()

score = compute_score_with_inverted_index(test_features, inverted_index, len(image_paths))
rank_ID = np.argsort(-score)

# 不进行反馈显示初始检索结果
if not enable_feedback:
    print("Initial Search Results")
    top_10 = min(len(rank_ID), 10)
    for i in range(top_10):
        rank = i + 1
        img_id = rank_ID[i]
        img_score = score[img_id]
        img_path = image_paths[img_id]
        print("Rank #{0}: {1} (score: {2:.6f})".format(rank, img_path, img_score))

# 进行反馈迭代优化查询向量
if enable_feedback:
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        print("Relevance Feedback - Round {}".format(iteration + 1))
        
        positive_ids, negative_ids = collect_feedback_from_ids(
            rank_ID, image_paths, im_features, top_k=top_k
        )
        
        
        updated_query = apply_feedback(
            original_query_vector if iteration == 0 else test_features,
            positive_ids, 
            negative_ids, 
            im_features
        )
        
        test_features = updated_query
        
        score = compute_score_with_inverted_index(test_features, inverted_index, len(image_paths))
        rank_ID = np.argsort(-score)
        
        # 显示当前迭代的检索结果
        print("Updated Search Results")
        top_10 = min(len(rank_ID), 10)
        for i in range(top_10):
            rank = i + 1
            img_id = rank_ID[i]
            img_score = score[img_id]
            img_path = image_paths[img_id]
            print("Rank #{0}: {1} (score: {2:.6f})".format(rank, img_path, img_score))
        
        print("\nContinue feedback? (y/n): ")
        continue_feedback = raw_input().strip().lower()
        if continue_feedback != 'y':
            break
        
        iteration += 1

# 进行RANSAC验证
ransac_results = ransac_verification(
    query_img=im,
    query_kpts=kpts,
    query_des=des,
    query_word_indices=query_word_indices,
    rank_ID=rank_ID,
    image_paths=image_paths,
    voc_tree=voc_tree,
    fea_det=fea_det,
    top_k=10
)

# 显示RANSAC验证结果
display_ransac_results(ransac_results, top_n=3)
