# -*- coding: utf-8 -*-

import numpy as np
from scipy.cluster.vq import kmeans, vq
import pickle


class VocabularyTreeNode:
    """词汇树节点类"""
    
    def __init__(self, level=0, max_level=0, k=10):
        self.level = level
        self.max_level = max_level
        self.k = k
        self.centers = None
        self.children = []
        self.is_leaf = (level == max_level)
        self.leaf_index = None
        
    def build(self, descriptors):
        """递归构建词汇树节点及其子树"""
        if self.is_leaf:
            if len(descriptors) < self.k:
                self.centers = descriptors
                self.k = len(descriptors)
            else:
                self.centers, _ = kmeans(descriptors, self.k, 1)
            return
        
        self.centers, _ = kmeans(descriptors, self.k, 1)
        words, _ = vq(descriptors, self.centers)
        
        self.children = [VocabularyTreeNode(self.level + 1, self.max_level, self.k) 
                        for _ in range(self.k)]
        
        for i in range(self.k):
            cluster_descriptors = descriptors[words == i]
            if len(cluster_descriptors) > 0:
                self.children[i].build(cluster_descriptors)
    
    def quantize_batch(self, descriptors):
        """批量量化特征描述符到视觉单词索引"""
        if self.is_leaf:
            if self.k == 1:
                return np.full(len(descriptors), self.leaf_index, dtype=np.int32)
            else:
                words, _ = vq(descriptors, self.centers)
                return np.array([self.leaf_index + w for w in words], dtype=np.int32)
        
        words, _ = vq(descriptors, self.centers)
        result = np.zeros(len(descriptors), dtype=np.int32)
        
        for i in range(self.k):
            mask = (words == i)
            if np.any(mask):
                result[mask] = self.children[i].quantize_batch(descriptors[mask])
        
        return result
    
    def get_num_leaves(self):
        """计算以当前节点为根的子树中叶节点对应的视觉单词总数"""
        if self.is_leaf:
            return self.k
        
        count = 0
        for child in self.children:
            count += child.get_num_leaves()
        return count
    
    def assign_leaf_indices(self, start_index=0):
        """为叶节点分配全局视觉单词索引"""
        if self.is_leaf:
            self.leaf_index = start_index
            return start_index + self.k
        
        current_index = start_index
        for child in self.children:
            current_index = child.assign_leaf_indices(current_index)
        return current_index


class VocabularyTree:
    """词汇树类，实现基于层次聚类的视觉词汇树"""
    
    def __init__(self, depth=3, k=10):
        self.depth = depth
        self.k = k
        self.root = None
        self.num_leaves = 0
        
    def build(self, descriptors):
        """构建词汇树"""
        self.root = VocabularyTreeNode(level=0, max_level=self.depth-1, k=self.k)
        self.root.build(descriptors)
        
        self.num_leaves = self.root.get_num_leaves()
        self.root.assign_leaf_indices(0)
        
    def quantize(self, descriptors):
        """量化特征描述符到视觉单词索引"""
        return self.root.quantize_batch(descriptors)
    
    def save(self, filepath):
        """保存词汇树模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'depth': self.depth,
                'k': self.k,
                'root': self.root,
                'num_leaves': self.num_leaves
            }, f)
    
    @staticmethod
    def load(filepath):
        """从文件加载词汇树模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            tree = VocabularyTree(depth=data['depth'], k=data['k'])
            tree.root = data['root']
            tree.num_leaves = data['num_leaves']
            return tree
