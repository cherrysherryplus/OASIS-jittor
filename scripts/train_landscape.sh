# 训练集9600 验证集400 测试集1000 
# 每一个epoch输出一次结果，每2个epoch保存一次latest 每10epoch保存一次ckpt
python train.py --name oasis_landscape --dataset_mode landscape --gpu_ids 0 \
--dataroot ./datasets/landscape --batch_size 16 --num_epochs 200 --no_3dnoise \
--freq_print 600 --freq_save_latest 1200 --freq_save_ckpt 6000 \
--freq_smooth_loss 60 --freq_save_loss 600 \
--freq_fid 1200 \
--param_free_norm batch \
--continue_train \
--which_iter latest