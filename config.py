import argparse
from email.policy import default
import pickle
import os
import utils.utils as utils


def read_arguments(train=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    opt = parser.parse_args()
    if train:
        set_dataset_default_lm(opt, parser)
        if opt.continue_train:
            update_options_from_file(opt, parser)
    opt = parser.parse_args()
    opt.phase = 'train' if train else 'test'
    if train:
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
    utils.fix_seed(opt.seed)
    print_options(opt, parser)
    if train:
        save_options(opt, parser)
    return opt


def add_all_arguments(parser, train):
    #--- general options ---
    parser.add_argument('--name', type=str, default='label_landscape', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    ## gpu_ids
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true',default=False, help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    ## 默认使用计图的数据集 landscape
    parser.add_argument('--dataroot', type=str, default='/home/oasis_jittor/models/Sample-And-Split-main/landscape_subset', help='path to dataset root')
    parser.add_argument('--dataset_mode', type=str, default='landscape', help='this option indicates which dataset should be loaded')
    
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=64, help='# of gen filters in first conv layer in generator')

    ## 默认使用batchnorm，因为一般是单卡训练，虽然syncbn的效果更好
    parser.add_argument('--param_free_norm', type=str, default='batch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--no_3dnoise', action='store_true', default=False, help='if specified, do *not* concatenate noise to label maps')
    parser.add_argument('--z_dim', type=int, default=64, help="dimension of the latent z vector")
    parser.add_argument('--use_DPM', action='store_true', default=True,
                        help='whether to use double pooling module')
    parser.add_argument('--norm_mod', action='store_true', default=False,
                        help='decide to use spade or spade_light(clade)')

    # freq_*等参数，根据训练集数量和batch_size计算结果来设置
    if train:
        parser.add_argument('--freq_print', type=int, default=1000, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=20000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2500, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=800, help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--channels_D', type=int, default=64, help='# of discrim filters in first conv layer in discriminator')
        parser.add_argument('--add_vgg_loss', action='store_true',default=True, help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, help='weight for LabelMix regularization')

        parser.add_argument('--use_fid_inception', action='store_true', default=False, help='if specified, use fid inception instead of normal inception v3')
    else:
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves testing results here.')
        parser.add_argument('--ckpt_iter', type=str, default='best', help='which epoch to load to evaluate a model')
        # test_only在正式测试时使用，表示使用A榜评测数据进行测试，而不是使用划分的验证集进行验证
        parser.add_argument('--test_only', action='store_true', default=False, help='if specified, load dataset specified for test')
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset_mode == "ade20k":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
    if opt.dataset_mode == "cityscapes":
        parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lambda_labelmix=5.0)
        parser.set_defaults(freq_fid=2500)
        parser.set_defaults(EMA_decay=0.999)
    # jittor landscape
    if opt.dataset_mode == "landscape":
        parser.set_defaults(lr_g=0.0004)
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.999)
    if opt.dataset_mode == "coco":
        parser.set_defaults(lambda_labelmix=10.0)
        parser.set_defaults(EMA_decay=0.9999)
        parser.set_defaults(num_epochs=100)


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir,opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
