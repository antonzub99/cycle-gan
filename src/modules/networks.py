import torch
import torch.nn as nn


def init_weights(m):
    classname = m.__class__.__name__
    init_gain = 0.02
    if hasattr(m, 'weight') and (classname.find('Conv') !=-1):
        nn.init.normal_(m.weight.data, 0.0, init_gain)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)


class ResnetBlock(nn.Module):
    """

    """
    def build_resblock(self, dim, use_bias=True, use_dropout=True):
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True)
        ]
        if use_dropout:
            layers.append(
                nn.Dropout(0.5)
            )
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False)
        ]
        block = nn.Sequential(*layers)
        return block

    def __init__(self, dim, use_bias=True, use_dropout=True):
        super().__init__()
        self.resblock = self.build_resblock(dim, use_bias=use_bias, use_dropout=use_dropout)

    def forward(self, x):
        x1 = x + self.resblock(x)
        return x1


class Generator(nn.Module):
    """

    """

    def build_gen(self, in_ch, out_ch, ngf=64, n_blocks=9, use_bias=True, use_dropout=True):
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.InstanceNorm2d(ngf, affine=False, track_running_stats=False),
            nn.ReLU(True)
        ]
        for i in range(2):
            layers += [
                nn.Conv2d(ngf * (2 ** i), ngf * (2 ** (i+1)),
                          kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.InstanceNorm2d(ngf * (2 ** (i+1)), affine=False, track_running_stats=False),
                nn.ReLU(True)
            ]
        for i in range(n_blocks):
            layers += [ResnetBlock(ngf * (2 ** 2),
                                   use_bias=use_bias, use_dropout=use_dropout)]
        for i in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * (2 ** (2 - i)), int(ngf * (2 ** (2-i) / 2)),
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                nn.InstanceNorm2d(int(ngf * (2 ** (2-i) / 2)), affine=False,
                                  track_running_stats=False),
                nn.ReLU(True)
            ]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        block = nn.Sequential(*layers)
        return block

    def __init__(self, in_ch, out_ch, ngf=64, n_blocks=9, use_bias=True, use_dropout=True):
        super().__init__()
        self.model = self.build_gen(in_ch, out_ch, ngf=ngf, n_blocks=n_blocks,
                                    use_bias=use_bias, use_dropout=use_dropout)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """

    """
    def build_disc(self, in_ch, ndf=64, n_layers=3, use_bias=True):
        layers = [
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        for i in range(1, n_layers+1):
            layers += [
                nn.Conv2d(ndf * (2 ** (i-1)), ndf * (2 ** i), kernel_size=4,
                          stride=2, padding=1, bias=use_bias),
                nn.InstanceNorm2d(ndf * (2 ** i), affine=False,
                                  track_running_stats=False),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv2d(ndf * (2 ** n_layers), 1, kernel_size=4, stride=1, padding=1)]
        block = nn.Sequential(*layers)
        return block

    def __init__(self, in_ch, ndf=64, n_layers=3, use_bias=True):
        super().__init__()
        self.model = self.build_disc(in_ch, ndf=ndf, n_layers=n_layers, use_bias=use_bias)

    def forward(self, x):
        return self.model(x)
