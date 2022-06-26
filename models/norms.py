import jittor.nn as nn
import jittor.linalg as lg
import jittor as jt
# TODO 谱归一化
# import torch.nn.utils.spectral_norm as spectral_norm
from models.spectral_norm import SpectralNorm


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return nn.Identity()
    else:
        return SpectralNorm


# 计图支持多卡并行的bn，但不支持syncbn
def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=None)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=True)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)

class SPADELight(nn.Module):
    """
    采用CLADE模块的改版SPAE，参数量大幅减少，计算复杂度下降
    """
    def __init__(self, opt, norm_nc, label_nc, no_instance=True, add_dist=False):
        super().__init__()
        self.no_instance = no_instance
        self.add_dist = add_dist
        # assert config_text.startswith('spade')
        # parsed = re.search('spade(\D+)(\d)x\d', config_text)
        # param_free_norm_type = str(parsed.group(1))
        # ks = int(parsed.group(2))

        # if param_free_norm_type == 'instance':
        #     self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'batch':
        #     self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % param_free_norm_type)
        self.param_free_norm = get_norm_layer(opt, norm_nc)
        self.class_specified_affine = ClassAffine(label_nc, norm_nc, add_dist)

        # if not no_instance:
        #     self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def execute(self, x, segmap, input_dist=None):

        # Part 1. 基础的归一化操作
        normalized = self.param_free_norm(x)

        # Part 2. scale 类别掩码图
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')

        # if not self.no_instance:
        #     inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
        #     segmap = segmap[:,:-1,:,:]

        # Part 3. 类别归一化
        out = self.class_specified_affine(normalized, segmap, input_dist)

        # if not self.no_instance:
        #     inst_feat = self.inst_conv(inst_map)
        #     out = torch.cat((out, inst_feat), dim=1)

        return out

class ClassAffine(nn.Module):
    """
    CLADE归一化，通过类别自适应归一化，降低SPAED里对于空间归一化的计算复杂度。
    """
    def __init__(self, label_nc, affine_nc, add_dist=False):
        super(ClassAffine, self).__init__()
        self.add_dist = add_dist
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        # self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        # self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        # nn.init.uniform_(self.weight)
        # nn.init.zeros_(self.bias)
        # if add_dist:
        #     self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        #     nn.init.zeros_(self.dist_conv_w.weight)
        #     nn.init.zeros_(self.dist_conv_w.bias)
        #     self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        #     nn.init.zeros_(self.dist_conv_b.weight)
        #     nn.init.zeros_(self.dist_conv_b.bias)

    # def affine_gather(self, input, mask):
    #     n, c, h, w = input.shape
    #     # process mask
    #     mask2 = jt.argmax(mask, 1) # [n, h, w]
    #     mask2 = mask2.view(n, h*w).long() # [n, hw]
    #     mask2 = mask2.unsqueeze(1).expand(n, self.affine_nc, h*w) # [n, nc, hw]
    #     # process weights
    #     weight2 = jt.unsqueeze(self.weight, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
    #     bias2 = jt.unsqueeze(self.bias, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
    #     # torch gather function
    #     class_weight = jt.gather(weight2, 0, mask2).view(n, self.affine_nc, h, w)
    #     class_bias = jt.gather(bias2, 0, mask2).view(n, self.affine_nc, h, w)
    #     return class_weight, class_bias

    # def affine_einsum(self, mask):
    #     class_weight = lg.einsum('ic,nihw->nchw', self.weight, mask)
    #     class_bias = lg.einsum('ic,nihw->nchw', self.bias, mask)
    #     return class_weight, class_bias

    def affine_embed(self, mask):
        arg_mask, _ = jt.argmax(mask, 1) # [n, h, w]
        embedding_weigth = nn.Embedding(self.label_nc, self.affine_nc)
        embedding_bias = nn.Embedding(self.label_nc, self.affine_nc)
        class_weight = embedding_weigth(arg_mask.long()).permute(0, 3, 1, 2) # [n, c, h, w]
        class_bias = embedding_bias(arg_mask.long()).permute(0, 3, 1, 2) # [n, c, h, w]
        return class_weight, class_bias

    def execute(self, input, mask, input_dist=None):
        # class_weight, class_bias = self.affine_gather(input, mask)
        # class_weight, class_bias = self.affine_einsum(mask) 
        class_weight, class_bias = self.affine_embed(mask)
        # if self.add_dist:
        #     input_dist = nn.interpolate(input_dist, size=input.size()[2:], mode='nearest')
        #     class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
        #     class_bias = class_bias * (1 + self.dist_conv_b(input_dist))
        x = input * class_weight + class_bias
        return x