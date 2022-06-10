#### 目录结构
Sample-And-Split
```bash
    landscape/
        --train
        ----imgs
        ----labels
        --val
        ----imgs
        ----labels
    main.py
    calc_classes_subset.py
    ...
    ...
    mmd_numpy_sklearn.py
    split_dataset.py
```
#### 使用
在Sample-And-Split目录下运行main.py，生成名为landscape_subset的数据集子集（包括训练集+验证集）。`total`表示子集数量，`val_num`表示子集中验证集数量。