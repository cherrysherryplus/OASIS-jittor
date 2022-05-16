# 和训练保持一致，不额外使用3d-noise作为输入，使用bn而不是sync-bn
# 新建了一个LandscapeTestDataset类，专用于官方测试数据
python test.py --name oasis_landscape --dataset_mode landscapetest --gpu_ids 0 \
--dataroot ./datasets/landscape --batch_size 16 \
--no_3dnoise \
--param_free_norm batch \
--test_only
