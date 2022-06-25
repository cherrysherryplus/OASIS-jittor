import jittor as jt
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_jittor
import config


if __name__ == '__main__':
    #--- read options ---#
    opt = config.read_arguments(train=True)
    opt.dataroot = "./datasets/sample_images"
    opt.norm_mod = True
    opt.num_epochs = 2
    # 24g p40 也不能把batch调到 8
    opt.batch_size = 1
    opt.freq_fid = 4
    opt.freq_print = 1
    # opt.freq_save_loss = 1
    # opt.no_spectral_norm = True
    
    #--- cuda ---#
    # export JT_SYNC=1
    # jt.flags.use_cuda = (jt.has_cuda and opt.gpu_ids!="-1")
    
    # jt.flags.amp_reg = jt.flags.amp_reg | 4
    # jt.flags.use_cuda_managed_allocator = 1
    # jt.flags.trace_py_var = 3
    # jt.flags.lazy_execution = 0
    
    # jt.cudnn.set_max_workspace_ratio(0.0)

    #--- create utils ---#
    timer = utils.timer(opt)
    visualizer_losses = utils.losses_saver(opt)
    losses_computer = losses.losses_computer(opt)
    dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
    im_saver = utils.image_saver(opt)
    fid_computer = fid_jittor(opt, dataloader_val)

    #--- create models ---#
    model = models.OASIS_model(opt)

    #--- create optimizers ---#
    optimizerG = jt.optim.Adam(model.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = jt.optim.Adam(model.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    #--- the training loop ---#
    already_started = False
    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
    for epoch in range(start_epoch, opt.num_epochs):
        for i, data_i in enumerate(dataloader):
            if not already_started and i < start_iter:
                continue
            already_started = True
            cur_iter = epoch*len(dataloader) + i
            print(cur_iter)
            image, label = models.preprocess_input(opt, data_i)
            
            #--- generator update ---#
            # jittor不会累积梯度
            loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
            loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
            optimizerG.zero_grad()
            optimizerG.backward(loss_G)
            optimizerG.step()
            # print(optimizerG.param_groups[0]['grads'][0].sum())

            #--- discriminator update ---#
            loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
            loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
            optimizerD.step(loss_D)

            # print(f"loss_G: {loss_G}, loss_D: {loss_D}")

            #--- stats update ---#
            if not opt.no_EMA:
                utils.update_EMA(model, cur_iter, dataloader, opt)
            if cur_iter % opt.freq_print == 0:
                im_saver.visualize_batch(model, image, label, cur_iter)
                timer(epoch, cur_iter)
            if cur_iter % opt.freq_save_ckpt == 0:
                utils.save_networks(opt, cur_iter, model)
            if cur_iter % opt.freq_save_latest == 0:
                utils.save_networks(opt, cur_iter, model, latest=True)
            if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
                is_best = fid_computer.update(model, cur_iter)
                if is_best:
                    utils.save_networks(opt, cur_iter, model, best=True)
            visualizer_losses(cur_iter, losses_G_list+losses_D_list)

        # 在每个epoch结束后，让Jittor强制同步（！回收内存）
        print(f"{epoch} epoch end~~~~~~")
        jt.sync_all()
        jt.gc()

    #--- after training ---#
    utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
    utils.save_networks(opt, cur_iter, model)
    utils.save_networks(opt, cur_iter, model, latest=True)
    is_best = fid_computer.update(model, cur_iter)
    if is_best:
        utils.save_networks(opt, cur_iter, model, best=True)

    print("The training has successfully finished")

