# modified from https://github.com/cosmic-cortex/pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256)):#, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet3D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, conv_depths[0], conv_depths[0]))
        encoder_layers.extend([Encoder3D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1])
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i])
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last3D(conv_depths[1], conv_depths[0], out_channels))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2])
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, kinematics, mesh, return_all=False):
        x_enc = [mesh]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1], kinematics)]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        if not return_all:
            return x_dec[-1]
        else:
            return x_enc + x_dec


def pad_to_shape(this, shp):
    """
    Pads this image with zeroes to shp.
    Args:
        this: image tensor to pad
        shp: desired output shape
    Returns:
        Zero-padded tensor of shape shp.
    """
    if len(shp) == 4:
        pad = (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    elif len(shp) == 5:
        pad = (0, shp[4] - this.shape[4], 0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2])
    return F.pad(this, pad)



class First3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder3D(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=downsample_kernel),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()

        layers1 = [
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
        ]
        layers2 = [
            nn.Conv3d(middle_channels+7, out_channels+7, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels+7),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels+7, out_channels+7, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels+7),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels+7, out_channels+3, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels+3),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels+3, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.center1 = nn.Sequential(*layers1)
        self.center2 = nn.Sequential(*layers2)
        
    def forward(self, x, kinematics):
        x = self.center1(x)
        kinematics = kinematics.view(kinematics.size()[0], kinematics.size()[1],1,1,1)
        kinematics = kinematics.repeat(1,1,x.size()[2],x.size()[3],x.size()[4])
        x = torch.cat((x, kinematics), axis=1)
        x = self.center2(x)
        return x 


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout3d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
#            nn.Softmax(dim=1)
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)
