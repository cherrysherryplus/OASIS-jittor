# from inception_jitttor import InceptionV3
# model_inc = InceptionV3(use_fid_inception=False)

from jittor.models.inception import Inception3
model_inc = Inception3()

for k,v in model_inc.state_dict().items():
    print(k)