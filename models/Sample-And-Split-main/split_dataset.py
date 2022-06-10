from pathlib import Path
import os
import os.path as osp
import random
from shutil import move,copy

from tqdm import tqdm


def check_img_label_legal(img_path, label_path):
    imgs = sorted(img_path.iterdir())
    labels = sorted(label_path.iterdir())
    assert len(imgs) == len(labels), f"length of {img_path} is unequal to length of {label_path}"
    for img, label in zip(imgs, labels):
        image_name, label_name = os.path.splitext(osp.split(img)[1])[0], os.path.splitext(osp.split(label)[1])[0]
        assert image_name == label_name,\
            '%s and %s are not matching' % (img, label)


def create_subset(dataset_root, subset_root, filenames, subset):
    # 先将采样的全放入train
    dataset_train = dataset_root / "train"
    dataset_train_imgs = dataset_train / "imgs"
    dataset_train_labels = dataset_train / "labels"

    subset_train = subset_root / "train"
    subset_train_imgs = subset_train / "imgs"
    subset_train_labels = subset_train / "labels"
    
    subset_train_imgs.mkdir(parents=True, exist_ok=True)
    subset_train_labels.mkdir(parents=True, exist_ok=True)
    
    for idx in tqdm(subset):
        labelfilename = filenames[idx]
        imgfilename = filenames[idx][:-4] + ".jpg"
        # 从数据集复制到子集中
        src_label = dataset_train_labels / labelfilename
        dst_label = subset_train_labels / labelfilename
        copy(src_label, dst_label)
        
        src_img = dataset_train_imgs / imgfilename
        dst_img = subset_train_imgs / imgfilename
        copy(src_img, dst_img)
        
    check_img_label_legal(subset_train_imgs, subset_train_labels)
    print("create subset done")


def split_train_val(subset_root, val_num):
    # 再从train中分离出val
    trainimgs = subset_root / "train" / "imgs"
    trainlabels = subset_root / "train" / "labels"

    testimgs = subset_root / "val" / "imgs"
    testlabels = subset_root / "val" / "labels"
    testimgs.mkdir(parents=True, exist_ok=True)
    testlabels.mkdir(parents=True, exist_ok=True)

    # 需要传入sampled_imglist sampled_lablelist
    imglist = sorted(trainimgs.iterdir())
    labellist = sorted(trainlabels.iterdir())

    sampled_imglist = sorted(random.sample(imglist, val_num))
    sampled_lablelist = sorted(map(lambda x:trainlabels / (x.stem + '.png'), sampled_imglist))
    for i in range(val_num):
        assert sampled_imglist[i].stem == sampled_lablelist[i].stem
        
    target_imglist = sorted(map(lambda x:testimgs / x.name, sampled_imglist))
    target_labellist = sorted(map(lambda x:testlabels / x.name, sampled_lablelist))
    for i in range(val_num):
        assert sampled_lablelist[i].stem == target_labellist[i].stem, f"{sampled_lablelist[i].stem}, {target_labellist[i].stem}"

    for src,dst in zip(sampled_imglist,target_imglist):
        assert src.stem == dst.stem, f"{src.stem}, {dst.stem}"
        move(src, dst)
    for src,dst in zip(sampled_lablelist,target_labellist):
        assert src.stem == dst.stem, f"{src.stem}, {dst.stem}"    
        move(src, dst)
        
    check_img_label_legal(trainimgs, trainlabels)
    check_img_label_legal(testimgs, testlabels)
    print("split train and val dataset from subset done")
    
    