import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.path = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.path(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()

        path = [
            nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        ]

        nfilters = 1
        nfilters_prev = 1
        for i in range(1, n_layers + 1):
            nfilters_prev = nfilters
            nfilters = min(2**i, 8)
            path += [
                nn.Conv2d(
                    hidden * nfilters_prev,
                    hidden * nfilters,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden * nfilters),
                nn.SiLU(),
            ]

        path += [
            nn.Conv2d(
                hidden * nfilters, out_channels, kernel_size=4, stride=1, padding=1
            )
        ]

        self.path = nn.Sequential(*path)

    def forward(self, x):
        x = self.path(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden: int = 128,
        n_blocks: int = 3,
    ):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, hidden, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(True),
        ]

        # downsampling
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            model += [
                nn.Conv2d(
                    hidden * mult,
                    hidden * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden * mult * 2),
                nn.SiLU(True),
            ]
            mult *= 2

        # residual blocks
        for _ in range(n_blocks):
            model += [ResBlock(hidden * mult)]

        # upsampling
        for i in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(
                    hidden * mult,
                    int(hidden * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(int(hidden * mult / 2)),
                nn.SiLU(True),
            ]
            mult = int(mult / 2)

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(hidden, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.path = nn.Sequential(*model)

    def forward(self, x):
        x = self.path(x)
        return x


if __name__ == "__main__":
    img_shape = (1, 3, 256, 128)
    img = torch.ones(img_shape)

    gen = Generator(in_channels=3, out_channels=3, hidden=32)
    yhat = gen(img)

    assert img.size() == yhat.size()

    dis = Discriminator(3, 1, 64, 3)
    print(dis(yhat))
