import jittor.nn as nn
import jittor as jt
from models.vgg_losses import VGG19

class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = nn.MSELoss()

    def loss(self, input, label, for_real):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        loss = nn.cross_entropy_loss(input, target, reduction='none')
        if for_real:
            loss = jt.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = jt.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        loss = self.labelmix_function(mixed_D_output, output_D_mixed)
        return loss


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = jt.sum(label, dims=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = (1.0 / class_occurence) * label.numel() / (num_of_classes * label.shape[1])
        integers, _ = jt.argmax(label, dim=1, keepdims=True)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = jt.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers, _ = jt.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = jt.clamp(integers, min_v=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, input, target_is_real):
    with jt.no_grad():
        if target_is_real:
            return jt.float32(1).expand_as(input)
        else:
            return jt.float32(0).expand_as(input)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
