import numpy as np
import random
import pandas as pd
from pathlib import Path


def legal_check(subset, subset_size):
    t = np.array(subset)
    _,cs = np.unique(t, return_counts=True)
    return len(subset)==subset_size and np.all(cs==1)


def sample_subset(cluster_results, filenames, total, subset_path):
    # sample from original dataset
    subset = []
    major_labels = []
    major_subset = []
    minor_labels = []
    minor_subset = []
    subset_size = total
    print(f"total subset length: {total}")

    # compute index ranges
    labels, counts = np.unique(cluster_results, return_counts=True)
    prefix_sum = [0]*(labels.size+1)
    for i in range(1, len(prefix_sum)):
        prefix_sum[i] = prefix_sum[i-1] + counts[i-1]    
    print(f"index ranges: {prefix_sum}")

    # <100的全都要
    for idx,(label,count) in enumerate(zip(labels, counts)):
        if count <= 100:
            l = list(range( prefix_sum[idx], prefix_sum[idx+1]))
            minor_subset += l
            minor_labels.append(label)
            total -= count
    
    # 区分聚类结果中的多数类，少数类
    for label in labels:
        if label not in minor_labels:
            major_labels.append(label)
    print(f"major_clusters: {major_labels}\nminor_clusters: {minor_labels}")

    # major labels's sample ratios
    ratio = counts[major_labels] / counts[major_labels].sum()
    print(f"sample major clusters with ratio: {ratio}")

    # 按比例划分每个多数类需要采样的数量
    d = {}
    rest = total
    for idx,(label,r) in enumerate(zip(major_labels, ratio)):
        if idx < ratio.size-1:
            d[label] = int(total * r)
            rest = rest-d[label]
            # print(d[label], rest)
        else:
            d[label] = rest

    for label,sample_num in d.items():
        l = random.sample(range(prefix_sum[label], prefix_sum[label+1]), sample_num)
        major_subset += l

    subset = major_subset + minor_subset
    assert legal_check(subset, subset_size),"check failed"
    np.save(subset_path, subset)
    print("sample subset done")


if __name__ == "__main__":
    mmd_cluster = "labels_mmd.npy"
    ed_cluster = "labels_ed.npy"
    subset_path = "subset.npy"
    total = 3200
    filenames = pd.read_csv(Path("train.csv"), index_col=0).index

    # 使用ed距离聚类可以将mmd_cluster改为ed_cluster
    cluster_results = np.load(mmd_cluster)
    sample_subset(cluster_results, filenames, total, subset_path)

