"""
Official EDSR/RCAN Model Stubs — Pickle compatibility for torch.load.

Extracted from model_manager.py. Registers fake ``model.*`` modules so
``torch.load`` can unpickle full model objects saved by the official
EDSR-PyTorch / RCAN repos.
"""

import logging
import sys
import types

import torch
import torch.nn as nn

log = logging.getLogger("videoforge")

_STUBS_REGISTERED = False


def register_official_model_stubs() -> None:
    """
    Register fake ``model.*`` modules so ``torch.load`` can unpickle full model
    objects saved by the official EDSR-PyTorch / RCAN repos.

    These repos save the entire ``nn.Module`` with ``torch.save(model, path)``,
    which records class paths like ``model.rcan.RCAN``.  We provide compatible
    stub classes that pickle can instantiate.
    """
    global _STUBS_REGISTERED
    if _STUBS_REGISTERED:
        return
    _STUBS_REGISTERED = True

    import math

    # ── model.common ──────────────────────────────────────────────────
    common = types.ModuleType("model.common")

    def default_conv(in_ch, out_ch, kernel_size, bias=True):
        return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bias)

    common.default_conv = default_conv

    class MeanShift(nn.Conv2d):
        def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040),
                     rgb_std=(1.0, 1.0, 1.0), sign=-1):
            super().__init__(3, 3, kernel_size=1)
            std = torch.Tensor(rgb_std)
            self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
            self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
            for p in self.parameters():
                p.requires_grad = False

    common.MeanShift = MeanShift

    class Upsampler(nn.Sequential):
        def __init__(self, conv=default_conv, scale=4, n_feat=64,
                     bn=False, act=False, bias=True):
            m = []
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv(n_feat, 4 * n_feat, 3, bias))
                    m.append(nn.PixelShuffle(2))
            elif scale == 3:
                m.append(conv(n_feat, 9 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(3))
            super().__init__(*m)

    common.Upsampler = Upsampler

    class ResBlock(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     bias=True, bn=False, act=None, res_scale=1):
            super().__init__()
            m = [conv(n_feat, n_feat, kernel_size, bias=bias),
                 nn.ReLU(True),
                 conv(n_feat, n_feat, kernel_size, bias=bias)]
            self.body = nn.Sequential(*m)
            self.res_scale = res_scale

        def forward(self, x):
            return x + self.body(x) * self.res_scale

    common.ResBlock = ResBlock

    # ── model.rcan ────────────────────────────────────────────────────
    rcan_mod = types.ModuleType("model.rcan")

    class CALayer(nn.Module):
        """Channel Attention with fc1/relu1(PReLU)/fc2 structure."""
        def __init__(self, channel=64, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
            self.relu1 = nn.PReLU(channel // reduction)
            self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.fc1(y)
            y = self.relu1(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            return x * y

    class SALayer(nn.Module):
        """Spatial Attention with a single 7x7 conv."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, 7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            y = self.conv1(avg_out)
            y = self.sigmoid(y)
            return x * y

    class CSAM(nn.Module):
        """Combined Channel + Spatial Attention Module."""
        def __init__(self, channel=64, reduction=16):
            super().__init__()
            self.ca = CALayer(channel, reduction)
            self.sa = SALayer()

        def forward(self, x):
            x = self.ca(x)
            x = self.sa(x)
            return x

    rcan_mod.CALayer = CALayer

    class RCAB_Stub(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     reduction=16, bias=True, bn=False, act=None, res_scale=1):
            super().__init__()
            modules = [conv(n_feat, n_feat, kernel_size, bias=bias),
                       act if act else nn.ReLU(True),
                       conv(n_feat, n_feat, kernel_size, bias=bias),
                       CSAM(n_feat, reduction)]
            self.body = nn.Sequential(*modules)
            self.res_scale = res_scale

        def forward(self, x):
            return x + self.body(x) * self.res_scale

    rcan_mod.RCAB = RCAB_Stub

    class ResidualGroup_Stub(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     reduction=16, act=None, res_scale=1, n_resblocks=20):
            super().__init__()
            modules = [RCAB_Stub(conv, n_feat, kernel_size, reduction,
                                 act=act, res_scale=res_scale)
                       for _ in range(n_resblocks)]
            modules.append(conv(n_feat, n_feat, kernel_size))
            self.body = nn.Sequential(*modules)

        def forward(self, x):
            return x + self.body(x)

    rcan_mod.ResidualGroup = ResidualGroup_Stub

    class RCAN_Stub(nn.Module):
        def __init__(self, args=None, conv=default_conv):
            super().__init__()

        def forward(self, x):
            pass

    rcan_mod.RCAN = RCAN_Stub

    # ── model.edsr ────────────────────────────────────────────────────
    edsr_mod = types.ModuleType("model.edsr")

    class EDSR_Stub(nn.Module):
        def __init__(self, args=None, conv=default_conv):
            super().__init__()

        def forward(self, x):
            pass

    edsr_mod.EDSR = EDSR_Stub

    # ── Register in sys.modules ───────────────────────────────────────
    model_pkg = types.ModuleType("model")
    model_pkg.__path__ = []
    model_pkg.common = common
    model_pkg.rcan = rcan_mod
    model_pkg.edsr = edsr_mod

    sys.modules["model"] = model_pkg
    sys.modules["model.common"] = common
    sys.modules["model.rcan"] = rcan_mod
    sys.modules["model.edsr"] = edsr_mod

    log.info("Registered official EDSR/RCAN model stubs")
