#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import math
import joblib
from sklearn import preprocessing
from vocabulary_tree import VocabularyTree
from Inverted_index import compute_score_with_inverted_index
from RANSAC import ransac_verification, display_ransac_results
from Relevance_Feedback import collect_feedback_from_ids, apply_feedback


def show_top_results_grid(rank_ID, score, image_paths, top_n=10,
                          window_name="Top Search Results", query_img=None):
    """将前 top_n 检索结果拼成网格并可视化，可选同时显示查询图像（更美观的布局）"""
    top_n = min(len(rank_ID), top_n)
    top_ids = rank_ID[:top_n]

    # 文本输出
    for i, img_id in enumerate(top_ids):
        rank = i + 1
        img_score = score[img_id]
        img_path = image_paths[img_id]
        print("Rank #{0}: {1} (score: {2:.6f})".format(rank, img_path, img_score))

    # 读取并可视化
    vis_images = []

    # 先准备统一的缩放尺寸
    target_w, target_h = 320, 240

    # 查询图像放在第一个位置，便于对比
    if query_img is not None:
        q_h, q_w = query_img.shape[:2]
        if q_w > target_w or q_h > target_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        q_img = cv2.resize(query_img, (target_w, target_h), interpolation=interp)
        # 半透明顶部条 + "Query" 标题
        overlay = q_img.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 32), (0, 0, 0), -1)
        q_img = cv2.addWeighted(overlay, 0.4, q_img, 0.6, 0)
        cv2.putText(q_img,
                    "Query",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
        vis_images.append(q_img)

    # 检索结果缩略图
    for i, img_id in enumerate(top_ids):
        img = cv2.imread(image_paths[img_id])
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        if w0 > target_w or h0 > target_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        img = cv2.resize(img, (target_w, target_h), interpolation=interp)

        # 半透明顶部条 + Rank 标签
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 32), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.putText(img,
                    "Rank {}".format(i + 1),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        vis_images.append(img)

    if len(vis_images) == 0:
        print("Warning: no images loaded for visualization.")
        return

    cols = 5
    h, w = vis_images[0].shape[:2]
    margin = 12

    if query_img is not None:
        num_results = len(vis_images) - 1
        result_rows = int(math.ceil(max(num_results, 1) * 1.0 / cols))
        rows = 1 + result_rows  # 第 1 行给 Query
    else:
        num_results = len(vis_images)
        rows = int(math.ceil(num_results * 1.0 / cols))

    grid_h = rows * h + (rows + 1) * margin
    grid_w = cols * w + (cols + 1) * margin

    # 深灰背景
    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    for idx, img in enumerate(vis_images):
        if query_img is not None and idx == 0:
            # Query 居中放在第一行
            y = margin
            x = (grid_w - w) // 2
        else:
            if query_img is not None:
                res_idx = idx - 1
                r = res_idx // cols + 1  # 从第二行开始
            else:
                res_idx = idx
                r = res_idx // cols
            c = res_idx % cols
            y = margin + r * (h + margin)
            x = margin + c * (w + margin)
        grid[y:y + h, x:x + w, :] = img

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def search_by_features(features, voc_tree, inverted_index, im_features, idf, image_paths, num_results=10):
    """基于特征进行搜索"""
    # 量化特征到视觉词汇
    words = voc_tree.quantize(features)
    query_word_indices = np.array(words, dtype=np.int32)
    
    # 构建查询向量
    test_features = np.zeros((1, voc_tree.num_words), "float32")
    for w in query_word_indices:
        test_features[0][w] += 1
    
    # 计算TF-IDF
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')
    
    # 使用倒排索引计算得分
    score = compute_score_with_inverted_index(test_features, inverted_index, len(image_paths))
    rank_ID = np.argsort(-score)
    
    # 返回前num_results个结果
    results = []
    for i in range(min(num_results, len(rank_ID))):
        img_id = rank_ID[i]
        results.append((score[img_id], image_paths[img_id]))
    
    return results


def main():
    """主函数 - 命令行版本"""
    import argparse as ap
    
    parser = ap.ArgumentParser(description="图像检索系统 - 基于词汇树和倒排索引")
    parser.add_argument("-i", "--image", help="Path to query image", required="True")
    parser.add_argument("-f", "--feedback", action="store_true", help="Enable relevance feedback")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display for feedback")
    args = vars(parser.parse_args())
    image_path = args["image"]
    enable_feedback = args["feedback"]
    top_k = args["top-k"]

    im_features, image_paths, idf, numWords, voc_tree, inverted_index = joblib.load("tree-bag-of-words.pkl")

    fea_det = cv2.SIFT_create()

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

    # 不进行反馈：显示一次初始检索结果 + 可视化
    if not enable_feedback:
        print("Initial Search Results (Top 10)")
        show_top_results_grid(rank_ID, score, image_paths,
                              top_n=10,
                              window_name="Top 10 Search Results",
                              query_img=im)

    # 进行反馈迭代优化查询向量，并在每轮显示可视化
    if enable_feedback:
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            print("Relevance Feedback - Round {}".format(iteration + 1))

            # 在用户选择相关 / 非相关之前，可视化当前前 top_k 个检索结果
            # 这样用户可以直接根据图像编号进行标注
            show_top_results_grid(
                rank_ID,
                score,
                image_paths,
                top_n=top_k,
                window_name="Feedback Candidates - Round {} (Top {})".format(
                    iteration + 1, min(top_k, len(rank_ID))
                ),
                query_img=im,
            )

            positive_ids, negative_ids = collect_feedback_from_ids(
                rank_ID, image_paths, im_features, top_k=top_k
            )
            
            updated_query = apply_feedback(
                original_query_vector if iteration == 0 else test_features,
                positive_ids, 
                negative_ids, 
                im_features,
                image_paths
            )
            
            # 使用更新后的查询向量重新计算得分
            test_features = updated_query
            
            score = compute_score_with_inverted_index(test_features, inverted_index, len(image_paths))
            rank_ID = np.argsort(-score)
            
            print("Updated Search Results (Round {})".format(iteration + 1))
            show_top_results_grid(
                rank_ID,
                score,
                image_paths,
                top_n=10,
                window_name="Updated Results - Round {}".format(iteration + 1),
                query_img=im
            )
            
            iteration += 1

        # RANSAC验证
        # 在最后一次迭代后，对前几个结果进行RANSAC验证
        print("RANSAC verification for top 3 results...")
        for i in range(min(3, len(rank_ID))):
            img_id = rank_ID[i]
            result_image = cv2.imread(image_paths[img_id])
            matches = ransac_verification(des, result_image)
            print("RANSAC matches for rank #{}: {}".format(i + 1, matches))
            
            # 可视化RANSAC结果
            display_ransac_results(orig_query, result_image, matches)


if __name__ == '__main__':
    main()