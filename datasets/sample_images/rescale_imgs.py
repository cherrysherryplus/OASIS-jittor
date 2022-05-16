import PIL.Image as Image
from pathlib import Path

dataset = "landscape"
dataroot = Path.cwd() / "datasets" / dataset
train_imgs = dataroot / "train" / "imgs"
val_imgs = dataroot / "val" / "imgs"

for trainimg in train_imgs.iterdir():
    print(trainimg)
    img = Image.open(trainimg).convert("RGB")
    ow,oh = img.size
    resized_img = img.resize((1024, 768), resample=Image.BILINEAR)
    resized_img.save(trainimg)
    
for valimg in val_imgs.iterdir():
    print(valimg)
    img = Image.open(valimg).convert("RGB")
    resized_img = img.resize((1024, 768), resample=Image.BILINEAR)
    resized_img.save(valimg)