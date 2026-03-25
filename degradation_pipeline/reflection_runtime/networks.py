from __future__ import annotations

import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str = "instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    if norm_type == "none":
        return None
    raise NotImplementedError(f"Normalization layer [{norm_type}] is not supported.")


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim,
            padding_type=padding_type,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            use_bias=use_bias,
        )

    def _build_conv_block(
        self,
        dim: int,
        *,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        conv_block = []
        padding = 0

        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] is not supported.")

        conv_block.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True),
            ]
        )
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        padding = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            padding = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] is not supported.")

        conv_block.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                norm_layer(dim),
            ]
        )
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        *,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model.extend(
                [
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            )

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model.append(
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            )

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            )

        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Sigmoid(),
            ]
        )
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_generator(
    *,
    input_nc: int = 6,
    output_nc: int = 3,
    ngf: int = 64,
    which_model_netG: str = "resnet_9blocks",
    norm: str = "instance",
    use_dropout: bool = False,
) -> ResnetGenerator:
    norm_layer = get_norm_layer(norm)
    if norm_layer is None:
        raise ValueError("Reflection synthesis generator requires a normalization layer.")
    if which_model_netG == "resnet_9blocks":
        n_blocks = 9
    elif which_model_netG == "resnet_6blocks":
        n_blocks = 6
    else:
        raise NotImplementedError(f"Generator model [{which_model_netG}] is not supported.")
    return ResnetGenerator(
        input_nc,
        output_nc,
        ngf,
        norm_layer=norm_layer,
        use_dropout=use_dropout,
        n_blocks=n_blocks,
    )
