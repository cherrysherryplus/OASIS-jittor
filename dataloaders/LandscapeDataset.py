import random
from jittor import dataset,transform
import os
import os.path as osp
from PIL import Image


class LandscapeDataset(dataset.Dataset):
    # 原图大小是768 * 1024；load_size=512相当于读入低分辨率的原图像；load_size=1024相当于读入正常分辨率的图像
    def __init__(self, opt, for_metrics):
        super(LandscapeDataset, self).__init__()
        
        # test low resolution 192*256
        opt.load_size = 256
        opt.crop_size = 256
        # 默认29个类
        opt.label_nc = 29
        # TODO 先改为False，如果有需要再改回来
        opt.contain_dontcare_label = False
        # TODO contain_dontcare_label为False，semantic_nc就不用加一
        opt.semantic_nc = 29 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        # 宽高比：4:3
        opt.aspect_ratio = 1024 / 768

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

        # 计图中，__len__默认返回的是分批后的iter次数，即融合了torch DataLoader的功能；
        # 另外，使用total_len属性，表示数据集的真实大小
        self.set_attrs(**{"total_len":len(self.images)})


    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx])).convert('L')
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}
        
        
    # landscape数据集里面，没有划分验证集，这里暂时也不显式地从训练集划分出验证集；
    # 直接用train来训练，用val（A榜评测数据集）来测试。
    def list_images(self):
        # for_metrics的意义暂不明确，可以推理的是它的作用与 opt.phase=='test' 有联系
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"

        images = []
        # <landscape_dataset_dir> / train / {imgs, labels}
        path_img = os.path.join(self.opt.dataroot, mode, "imgs")
        for item in sorted(os.listdir(path_img)):
            images.append(item)
            
        labels = []
        path_lab = os.path.join(self.opt.dataroot, mode, "labels")
        for item in sorted(os.listdir(path_lab)):
            labels.append(item)

        # sanity check（完整性检查）
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            image_name, label_name = os.path.splitext(osp.split(images[i])[1])[0], os.path.splitext(osp.split(labels[i])[1])[0]
            assert image_name == label_name,\
                '%s and %s are not matching' % (images[i], labels[i])
                
        return images, labels, (path_img, path_lab)


    def transforms(self, image, label):
        assert image.size == label.size
        # resize (load_size不等于原图的大小，crop_size是在load_size基础上进行裁剪的)
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = transform.resize(image, (new_width, new_height), Image.BICUBIC)
        label = transform.resize(label, (new_width, new_height), Image.NEAREST)
        # flip（test阶段、for_metrics不翻转，显式指定no_flip也不翻转）
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = transform.hflip(image)
                label = transform.hflip(label)
        # to tensor（像RGB mode下的PIL.Image对象，会转换为FloatTensor，同时会缩放到[0,1.0]，所以后续label会乘上255）
        # normalize（只有图片要规范化）
        # image = transform.to_tensor(image) （二者选一即可，to_tensor和image_normalize/ImageNormalize）
        # image = transform.image_normalize(image, [0.5], [0.5])
        image = transform.ImageNormalize([0.5], [0.5])(image)
        label = transform.to_tensor(label)
        return image, label
