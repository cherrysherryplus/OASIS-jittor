from cProfile import label
import PIL.Image as Image
import numpy as np
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm


def calc_classes_distribution(label_dir, result_path="train.csv"):
    if isinstance(label_dir, Path):
        filenames = list(label_dir.iterdir())
    elif isinstance(label_dir, str):
        filenames = os.listdir(label_dir)
        
    data = {i:[0]*len(filenames) for i in range(29)}
    dists = pd.DataFrame(data, index=list(map(lambda x:x.name, filenames)))
    for filename in tqdm(filenames):
        img = np.array(Image.open(filename))
        counts = np.bincount(img.ravel())
        # http://c.biancheng.net/pandas/loc-iloc.html
        # .loc 是前闭后闭区间
        dists.loc[filename.name, 0:counts.size-1] = counts
    check_index_legal(dists)
    dists.to_csv(result_path)
    return dists


def get_classes_distribution(csv_path, label_dir=None, result_path="train.csv"):
    if csv_path is None:
        assert label_dir, "label_dir must not be None if csv_path is None"
        dists = calc_classes_distribution(label_dir, result_path)
    else:    
        dists = pd.read_csv(csv_path,index_col=0)
    print("calc or load classes distribution done")
    return dists


def check_index_legal(dists):
    for index, row in dists.head().iterrows():     
        row_lists=list(row)
        print(f"{index}\n{row_lists}")


if __name__ == "__main__":
    train_label_dir = Path("train/labels")
    dists = get_classes_distribution(None,label_dir = train_label_dir,result_path="train.csv")
    
    