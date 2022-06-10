import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize

from calc_classes_distribution import get_classes_distribution
from precompute_distances import precompute_distances
from cluster_scaled_data import cluster
from calc_sampled_subset import sample_subset
from split_dataset import create_subset, split_train_val


def main():
    # -------------------------------------
    # 设置随机数种子
    # -------------------------------------
    random.seed(0)
    
    # -------------------------------------
    # 子集数量
    # -------------------------------------
    total = 3400

    # -------------------------------------
    # 计算每张图片的语义类别分布
    # -------------------------------------
    train_csv_path = Path("train.csv")
    train_label_dir = Path("/root/autodl-tmp/landscape/train/labels")
    if train_csv_path.exists():
        dists = get_classes_distribution(train_csv_path, label_dir=train_label_dir)
    else:
        dists = get_classes_distribution(None, label_dir=train_label_dir, result_path=train_csv_path)
    dists_scaled = normalize(dists)
        
    # -------------------------------------
    # 预先计算距离矩阵
    # -------------------------------------
    distance_type = "ed"
    distances_matrix_path = Path(f"{distance_type}.npy")
    if not distances_matrix_path.exists():
        precompute_distances(distance_type, dists_scaled, distances_matrix_path)
    if distance_type == 'mmd':
        data_tobe_clustered = np.load(distances_matrix_path)
    else:
        data_tobe_clustered = dists_scaled

    # -------------------------------------
    # 层次聚类
    # -------------------------------------
    labels_path = Path("cluster_labels.npy")
    if distance_type == "mmd":
        affinity_option = 'precomputed'
    else:
        affinity_option = 'euclidean'
    if not labels_path.exists():
        cluster(affinity_option=affinity_option, \
                data=data_tobe_clustered, \
                labels_path=labels_path)
    
    cluster_results = np.load(labels_path)

    # -------------------------------------
    # 采样子集
    # -------------------------------------
    subset_path = Path("subset.npy")
    filenames = pd.read_csv(train_csv_path, index_col=0).index
    if not subset_path.exists():
        sample_subset(cluster_results, filenames, total, subset_path)
    subset = np.load(subset_path)

    # -------------------------------------
    # 划分到指定目录内（包括训练、验证集）
    # -------------------------------------
    val_num = 200
    dataset_root = Path("/root/autodl-tmp/landscape")
    subset_root = Path("landscape_subset")
    create_subset(dataset_root, subset_root, filenames, subset)
    split_train_val(subset_root, val_num)


if __name__ == "__main__":
    main()
    
    