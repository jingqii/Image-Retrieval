#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

"""
相关反馈模块：优化图像检索查询向量
"""

import numpy as np
from sklearn import preprocessing


def compute_word_frequency(feature_vectors):
    if len(feature_vectors) == 0:
        return None
    return np.mean(feature_vectors, axis=0)


def reweight_query_vector(original_query, positive_features, negative_features, 
                         alpha=0.7, beta=0.3, gamma=0.2):
    # 保留原始查询信息
    updated_query = original_query.copy() * alpha
    
    # 增加相关图像中出现的单词的权重
    if len(positive_features) > 0:
        positive_freq = compute_word_frequency(positive_features)
        for w in range(original_query.shape[1]):
            if positive_freq[w] > 0:
                updated_query[0, w] += beta * positive_freq[w]
    
    # 减少不相关图像中出现的单词的权重
    if len(negative_features) > 0:
        negative_freq = compute_word_frequency(negative_features)
        for w in range(original_query.shape[1]):
            if negative_freq[w] > 0:
                updated_query[0, w] = max(0, updated_query[0, w] - gamma * negative_freq[w])
    
    # 确保非负并归一化
    updated_query = np.maximum(updated_query, 0)
    updated_query = preprocessing.normalize(updated_query, norm='l2')
    return updated_query


def collect_feedback_from_ids(rank_ID, image_paths, im_features, top_k=20):
    positive_ids = []
    negative_ids = []
    
    print("Displaying top {} search results:".format(top_k))
    
    # 显示前k个结果
    display_ids = rank_ID[:top_k]
    for idx, img_id in enumerate(display_ids, start=1):
        print("{:2d}. {}".format(idx, image_paths[img_id]))
    
    # 收集相关图像标记
    print("\nPlease enter relevant image numbers (comma-separated, e.g., 1,3,5), press Enter to skip:")
    positive_input = raw_input("Relevant images: ").strip()
    
    if positive_input:
        try:
            positive_indices = [int(x.strip()) - 1 for x in positive_input.split(',')]
            positive_ids = [display_ids[i] for i in positive_indices if 0 <= i < len(display_ids)]
        except ValueError:
            print("Input format error, skipping relevant image marking")
    
    # 收集不相关图像标记
    print("\nPlease enter irrelevant image numbers (comma-separated), press Enter to skip:")
    negative_input = raw_input("Irrelevant images: ").strip()
    
    if negative_input:
        try:
            negative_indices = [int(x.strip()) - 1 for x in negative_input.split(',')]
            negative_ids = [display_ids[i] for i in negative_indices if 0 <= i < len(display_ids)]
        except ValueError:
            print("Input format error, skipping irrelevant image marking")
    
    return positive_ids, negative_ids


def apply_feedback(query_vector, positive_ids, negative_ids, im_features):
    positive_features = []
    negative_features = []
    
    # 收集正负样本特征
    for img_id in positive_ids:
        if img_id < len(im_features):
            positive_features.append(im_features[img_id])
    
    for img_id in negative_ids:
        if img_id < len(im_features):
            negative_features.append(im_features[img_id])
    
    # 有反馈时更新查询
    if len(positive_features) > 0 or len(negative_features) > 0:
        updated_query = reweight_query_vector(
            query_vector, 
            np.array(positive_features) if positive_features else np.array([]).reshape(0, query_vector.shape[1]),
            np.array(negative_features) if negative_features else np.array([]).reshape(0, query_vector.shape[1])
        )
        
        # 输出统计信息
        if len(positive_ids) > 0:
            print("\nMarked {} relevant images".format(len(positive_ids)))
        if len(negative_ids) > 0:
            print("Marked {} irrelevant images".format(len(negative_ids)))
        
        return updated_query
    else:
        print("\nNo valid feedback collected, using original query vector")
        return query_vector

