"""Plain 2D U-Net (Ronneberger et al.), the usual Noise2Noise backbone."""
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
    )


class UNet(nn.Module):
    """`depth` down/up levels, `base` channels at the top, doubled per level.
    Input height/width must be divisible by 2**depth (see `denoise` for padding)."""

    def __init__(self, in_ch=1, out_ch=1, base=32, depth=4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(in_ch, chs[0])] +
                                  [block(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(2 * chs[i], chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.down):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)
