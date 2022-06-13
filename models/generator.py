import jittor.nn as nn
import models.norms as norms
import jittor as jt
import torch.nn.functional as F

# F.relu_()


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img_fc = nn.Conv2d(ch*8, self.channels[-1], 3, padding=1)
        self.conv_img= nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        self.StripPooling = StripPooling(in_channels=128, pool_size=(20, 12), norm_layer=nn.BatchNorm2d)
        self.StripPooling2 = StripPooling(in_channels=256, pool_size=(20, 12), norm_layer=nn.BatchNorm2d)
        self.PyramidPooling = PyramidPooling(in_channels=64, norm_layer=nn.BatchNorm2d)

        for i in range(len(self.channels) - 1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i + 1], opt))
        if not self.opt.no_3dnoise:
            self.bio_norm = nn.Sequential(nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, ch, (3, 3), padding = 1), nn.BatchNorm2d(ch), nn.ReLU())
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, (3,3), padding=1)
        else:
            self.fc= nn.Conv2d(self.opt.semantic_nc, 16 * ch, (3,3), padding=1)

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2 ** (opt.num_res_blocks - 1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def execute(self, input, z=None):
        seg = input
        # if self.opt.gpu_ids != "-1":
        #     seg.cuda()
        if not self.opt.no_3dnoise:
            z = jt.randn(seg.size(0), self.opt.z_dim, dtype=jt.float32)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = jt.concat((z, seg), dim=1)
            dx = self.bio_norm(seg)
        x = nn.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        # dx = self.bio_norm(x)

        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.opt.num_res_blocks - 1:
                x = self.up(x)
        # print('x.shape',x.shape)
        x = x + dx

        if self.opt.use_DPM:
            x = self.PyramidPooling(x)
            # print('x.shape', x.shape)
            x = self.StripPooling(x)
            # print('x.shape', x.shape)
            x = self.StripPooling2(x)

            # print('x.shape', x.shape)
            x = self.conv_img_fc(x)
            x = self.conv_img(nn.leaky_relu(x,2e-1))
        else:
            x = self.conv_img(nn.leaky_relu(x, 2e-1))

        x = jt.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)

        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim
        if opt.norm_mod:
            self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
            self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
            if self.learned_shortcut:
                self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        else:
            self.norm_0 = norms.SPADELight(opt, fin, spade_conditional_input_dims)
            self.norm_1 = norms.SPADELight(opt, fmiddle, spade_conditional_input_dims)
            if self.learned_shortcut:
                self.norm_s = norms.SPADELight(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2)

    def execute(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out


# Square-shape Pooling Module of DPM from Layout-to-Image Translation with DPGAN
class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                   nn.LeakyReLU())
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def execute(self, x):
        _, _, h, w = x.size()
        feat1 = nn.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = nn.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = nn.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = nn.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return jt.concat((x, feat1, feat2, feat3, feat4), 1)

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                     nn.LeakyReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                     nn.LeakyReLU())
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                     nn.LeakyReLU())
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                     nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def execute(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = nn.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = nn.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = nn.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = nn.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(jt.nn.leaky_relu(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(jt.nn.leaky_relu(x2_5 + x2_4))
        out = self.conv3(jt.concat([x1, x2], dim=1))

        return jt.nn.leaky_relu(jt.concat((x, out), 1))