import jittor as jt
from jittor.misc import normalize
from typing import Any, Optional, TypeVar
import jittor.nn as nn
from jittor.nn import Module

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, dim=0, eps=1e-12):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.dim = dim
        if power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got power_iterations={}'.format(power_iterations))
        self.power_iterations = power_iterations
        self.eps = eps
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v):
        return v / (v.norm() + self.eps)
    
    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name)
        weight_mat = self.reshape_weight_to_matrix(w)

        for _ in range(self.power_iterations):
            v.assign(self.l2normalize((weight_mat.t() * u.unsqueeze(0)).sum(-1)))
            u.assign(self.l2normalize((weight_mat * v.unsqueeze(0)).sum(-1)))
        sigma = (u * (weight_mat * v.unsqueeze(0)).sum(-1)).sum()
        getattr(self.module, self.name).assign(w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name)
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        weight_mat = self.reshape_weight_to_matrix(w)
        height, width = weight_mat.shape

        u = jt.empty([height], dtype=w.dtype).gauss_(0, 1)
        v = jt.empty([width], dtype=w.dtype).gauss_(0, 1)
        u = self.l2normalize(u)
        v = self.l2normalize(v)

        setattr(self.module, self.name + "_u", u.stop_grad())
        setattr(self.module, self.name + "_v", v.stop_grad())

    def execute(self, *args):
        self._update_u_v()
        return self.module.execute(*args)
    
    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        w = getattr(module, name)
        if w is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        fn = SpectralNorm(module, name, n_power_iterations, dim, eps)
        module.register_pre_forward_hook(fn)
        return fn

T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (nn.ConvTranspose,
                               nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

