import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, basic_conv=None,
                 bias=False, downsample=False):
        super(Block, self).__init__()
        self.downsample = downsample
        self.fdc = nn.Sequential(
            basic_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                       dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(),
            # nn.ReflectionPad1d(dilation),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                      dilation=dilation, groups=groups, bias=bias),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.down = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            # nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=2, padding_mode='zeros', dilation=2),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample:
            x = self.down(x)
        left = self.left(x)
        right = self.fdc(x)
        out = left + right
        out = self.relu(out)

        return out


class AggEncoder(nn.Module):

    def __init__(self, basic_block=Block):
        super(AggEncoder, self).__init__()

        self.Encoder_0 = nn.Sequential(basic_block(1, 32, kernel_size=7, padding=3, basic_conv=nn.Conv1d))
        # Encoder
        self.Encoder_1 = nn.Sequential(basic_block(32, 48, basic_conv=nn.Conv1d, downsample=True))
        self.Encoder_2 = nn.Sequential(basic_block(48, 96, basic_conv=nn.Conv1d, downsample=True))
        self.Encoder_3 = nn.Sequential(basic_block(96, 144, basic_conv=nn.Conv1d, downsample=True))
        self.Encoder_4 = nn.Sequential(basic_block(144, 192, basic_conv=nn.Conv1d, downsample=True))
        self.Encoder_5 = nn.Sequential(basic_block(192, 240, basic_conv=nn.Conv1d, downsample=True))
        self.Encoder_6 = nn.Sequential(basic_block(240, 288, basic_conv=nn.Conv1d, downsample=True))

    def forward(self, origin):
        origin = origin.unsqueeze(1)
        # origin = torch.diff(origin)
        # origin = F.pad(origin, [1,0,0,0])

        # Encoder
        encoded0 = self.Encoder_0(origin)
        encoded1 = self.Encoder_1(encoded0)
        encoded2 = self.Encoder_2(encoded1)
        encoded3 = self.Encoder_3(encoded2)
        encoded4 = self.Encoder_4(encoded3)
        encoded5 = self.Encoder_5(encoded4)
        encoded6 = self.Encoder_6(encoded5)

        return encoded6


class AggDecoder(nn.Module):

    def __init__(self, basic_block=Block):
        super(AggDecoder, self).__init__()

        # Decoder
        self.Decoder_1 = nn.Sequential(
            nn.ConvTranspose1d(288, 288, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(288, 240, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(240, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(240, 192, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(192, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(192, 144, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(144, 144, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(144, 96, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(96, 48, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(48, 32, basic_conv=nn.Conv1d),
            nn.Conv1d(32, 1, kernel_size=1, padding=0),
        )
        self.Decoder_2 = nn.Sequential(
            nn.ConvTranspose1d(288, 288, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(288, 240, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(240, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(240, 192, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(192, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(192, 144, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(144, 144, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(144, 96, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(96, 48, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(48, 32, basic_conv=nn.Conv1d),
            nn.Conv1d(32, 1, kernel_size=1, padding=0),
        )
        self.Decoder_3 = nn.Sequential(
            nn.ConvTranspose1d(288, 288, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(288, 240, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(240, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(240, 192, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(192, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(192, 144, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(144, 144, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(144, 96, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(96, 48, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(48, 32, basic_conv=nn.Conv1d),
            nn.Conv1d(32, 1, kernel_size=1, padding=0),
        )
        self.Decoder_4 = nn.Sequential(
            nn.ConvTranspose1d(288, 288, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(288, 240, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(240, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(240, 192, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(192, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(192, 144, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(144, 144, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(144, 96, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(96, 48, basic_conv=nn.Conv1d),
            nn.ConvTranspose1d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            basic_block(48, 32, basic_conv=nn.Conv1d),
            nn.Conv1d(32, 1, kernel_size=1, padding=0),
        )


    def forward(self, embed):
        # Decoder
        decoded1 = self.Decoder_1(embed)
        decoded2 = self.Decoder_2(embed)
        decoded3 = self.Decoder_3(embed)
        decoded4 = self.Decoder_4(embed)
        out = torch.cat([decoded1, decoded2, decoded3, decoded4], dim=1)
        out = out.permute(0, 2, 1)
        return out
