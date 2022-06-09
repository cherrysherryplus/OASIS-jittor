from models.spectral_norm import spectral_norm
import jittor.nn as nn
import jittor.init as init
import jittor as jt

conv_0 = spectral_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1))

init.gauss_(conv_0.weight, 0, 1)
init.constant_(conv_0.bias, 0.0)

x = jt.rand(1,3,12,12)

print(conv_0(x))