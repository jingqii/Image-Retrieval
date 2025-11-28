#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np


def compute_score_with_inverted_index(query_features, inverted_index, num_docs):
    """使用倒排索引计算查询图像与所有训练图像的相似度得分"""
    score = np.zeros(num_docs, dtype='float32')
    query_words = np.nonzero(query_features[0])[0]
    
    for w in query_words:
        query_weight = query_features[0][w]
        if w < len(inverted_index) and len(inverted_index[w]) > 0:
            for doc_id, doc_weight in inverted_index[w]:
                score[doc_id] += query_weight * doc_weight
    
    return score
