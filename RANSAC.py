#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


def group_indices_by_word(word_indices):
    """将视觉单词索引按单词ID分组"""
    groups = {}
    for idx, word_id in enumerate(word_indices):
        groups.setdefault(word_id, []).append(idx)
    return groups


def match_with_descriptor_distance(query_des, query_word_ids, cand_des, cand_word_ids, 
                                   ratio_threshold=0.75, max_distance=200.0):
    """在共享视觉单词内进行描述符匹配并应用Lowe's ratio test过滤"""
    query_groups = group_indices_by_word(query_word_ids)
    cand_groups = group_indices_by_word(cand_word_ids)
    shared_words = set(query_groups.keys()) & set(cand_groups.keys())
    
    if not shared_words:
        return []
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = []
    
    for word_id in shared_words:
        q_idx = query_groups[word_id]
        c_idx = cand_groups[word_id]
        q_desc = query_des[q_idx]
        c_desc = cand_des[c_idx]
        
        knn_matches = bf.knnMatch(q_desc, c_desc, k=2)
        
        for pair in knn_matches:
            if len(pair) == 2:
                best, second = pair
                if best.distance < ratio_threshold * second.distance and best.distance < max_distance:
                    match = cv2.DMatch(q_idx[best.queryIdx], c_idx[best.trainIdx], best.distance)
                    matches.append(match)
    
    matches.sort(key=lambda x: x.distance)
    return matches


def ransac_verification(query_img, query_kpts, query_des, query_word_indices,
                       rank_ID, image_paths, voc_tree, fea_det, top_k=10):
    """使用RANSAC对检索结果进行空间几何验证"""
    min_inliers = 10
    ransac_threshold = 3.0
    max_iters = 10000
    ratio_threshold = 0.75
    max_distance = 200.0
    
    results = []
    
    for cand_id in rank_ID[:top_k]:
        cand_img = cv2.imread(image_paths[cand_id])
        cand_kpts, cand_des = fea_det.detectAndCompute(cand_img, None)
        
        if cand_des is None or len(cand_des) < 4:
            continue
        
        cand_words = voc_tree.quantize(cand_des)
        
        good_matches = match_with_descriptor_distance(
            query_des, query_word_indices, cand_des, cand_words,
            ratio_threshold=ratio_threshold, max_distance=max_distance
        )
        
        if len(good_matches) < max(8, min_inliers * 2):
            continue
        
        src_pts = np.float32([query_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([cand_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold, maxIters=max_iters)
        
        if mask is None:
            continue
        
        inliers = int(mask.ravel().sum())
        if inliers < min_inliers:
            continue
        
        inlier_ratio = float(inliers) / len(good_matches)
        if inlier_ratio < 0.2:
            continue
        
        vis = cv2.drawMatches(
            query_img, query_kpts,
            cand_img, cand_kpts,
            good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        results.append({
            "doc_id": cand_id,
            "path": image_paths[cand_id],
            "inliers": inliers,
            "ratio": inlier_ratio,
            "match_img": cand_img.copy(),
            "vis": vis
        })
    
    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results


# def display_ransac_results(results, top_n=5, show_vis=True):
#     """显示RANSAC验证结果"""
#     top_results = results[:top_n]
    
#     for idx, res in enumerate(top_results, start=1):
#         print("RANSAC result #{0}: {1} (inliers: {2}, ratio: {3:.3f})".format(
#             idx, res["path"], res["inliers"], res["ratio"]))
    
#     if show_vis:
#         top_3_results = results[:3]
#         for idx, res in enumerate(top_3_results, start=1):
#             vis_resized = cv2.resize(res["vis"], (1200, 600))
#             title = "Match #{0}: {1} (inliers: {2}, ratio: {3:.3f})".format(
#                 idx, os.path.basename(res["path"]), res["inliers"], res["ratio"])
#             cv2.namedWindow(title, cv2.WINDOW_NORMAL)
#             cv2.imshow(title, vis_resized)
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

