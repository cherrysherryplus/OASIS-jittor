import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import jittor as jt
import jittor.nn as nn
from jittor.nn import init
import numpy as np
import models.losses as losses


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            jt.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class OASIS_model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.netG = generators.OASIS_Generator(opt)
        if opt.phase == 'train':
            self.netD = discriminators.OASIS_Discriminator(opt)
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
        self.print_parameter_count()
        self.init_networks()
        # init_networks([self.netG,self.netD])
        with jt.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        self.load_checkpoints()


    def execute(self, image, label, mode, losses_computer):
        if mode == 'losses_G':
            loss_G = 0
            fake = self.netG(label)
            output_D = self.netD(fake)
            loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, [loss_G_adv, loss_G_vgg]

        if mode == 'losses_D':
            loss_D = 0
            with jt.no_grad():
                fake = self.netG(label)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(self.opt, label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                     output_D_fake,
                                                                                     output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]


        if mode == 'generate':
            with jt.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label)
                else:
                    fake = self.netEMA(label)
                return fake


    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(jt.load(path + "G.pkl"))
            else:
                self.netEMA.load_state_dict(jt.load(path + "EMA.pkl"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(jt.load(path + "G.pkl"))
            self.netD.load_state_dict(jt.load(path + "D.pkl"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(jt.load(path + "EMA.pkl"))


    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    # p.data时np ndarray, size属性是元素个数
                    param_count += sum([np.prod(p.shape) for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)


    def init_networks(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(weights_init_normal)


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    # 默认类型就是float32
    input_label = jt.zeros(shape=(bs, nc, h, w))
    input_semantics = input_label.scatter_(1, label_map, jt.array(1.0))
    return data['image'], input_semantics


def generate_labelmix(opt, label, fake_image, real_image):
    target_map, _ = jt.argmax(label, dim=1, keepdims=True)
    all_classes = jt.unique(target_map)
    for c in all_classes:
        # test
        import numpy as np
        mask_np = np.random.randint(0, 2, (1,))
        mask = jt.Var(mask_np)
        # original
        # mask = jt.randint(0, 2, (1,))
        target_map[target_map == c] = mask
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map

