import jittor as jt
import numpy as np
import random
import time
import os
import models.models as models
import matplotlib.pyplot as plt
from PIL import Image


# 设置随机数种子(python numpy jittor cupy)
# https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/_modules/jittor/misc.html#set_global_seed
def fix_seed(seed):
    jt.set_global_seed(seed)


# 计算开始的epoch和iter
def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter  = (start_iter + 1) %  dataset_size
    return start_epoch, start_iter


class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with jt.no_grad():
        state_EMA = model.netEMA.state_dict()
        state_G = model.netG.state_dict()
        for key in state_EMA.keys():
            state_EMA[key] = state_EMA[key] * opt.EMA_decay + state_G[key] * (1 - opt.EMA_decay)
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with jt.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label = models.preprocess_input(opt, data_i)
                fake = model.netEMA(label)
                num_upd += 1
                if num_upd > 50:
                    break


# 保存模型相关参数（可以包括模型、优化器、epoch、fid等信息，目前只存模型参数）
# https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/_modules/jittor.html#save
def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        jt.save(model.netG.state_dict(), path+'/%s_G.pkl' % ("latest"))
        jt.save(model.netD.state_dict(), path+'/%s_D.pkl' % ("latest"))
        if not opt.no_EMA:
            jt.save(model.netEMA.state_dict(), path+'/%s_EMA.pkl' % ("latest"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        jt.save(model.netG.state_dict(), path+'/%s_G.pkl' % ("best"))
        jt.save(model.netD.state_dict(), path+'/%s_D.pkl' % ("best"))
        if not opt.no_EMA:
            jt.save(model.netEMA.state_dict(), path+'/%s_EMA.pkl' % ("best"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        jt.save(model.netG.state_dict(), path+'/%d_G.pkl' % (cur_iter))
        jt.save(model.netD.state_dict(), path+'/%d_D.pkl' % (cur_iter))
        if not opt.no_EMA:
            jt.save(model.netEMA.state_dict(), path+'/%d_EMA.pkl' % (cur_iter))


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images")+"/"
        self.opt = opt
        self.num_cl = opt.label_nc + 1
        if opt.contain_dontcare_label:
            self.num_cl = opt.num_cl + 1
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with jt.no_grad():
            model.eval()
            fake = model.netG(label)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                fake = model.netEMA(label)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    print("label_tensor: ", label_tensor.shape)
    label_numpy = np.transpose(label_tensor.data, (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = jt.array(cmap[:num_cl])
    size = tens.size()
    color_image = jt.zeros((3, size[1], size[2]), jt.uint8)
    # jt.argmax会返回一个元组，第一个元素是索引，第二个元素是索引对应的值
    tens, _ = jt.argmax(tens, dim=0, keepdims=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0])
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

