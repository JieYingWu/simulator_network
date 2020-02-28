import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/fxia22/pointnet.pytorch
class PointNetfeat(nn.Module):
    def __init__(self, conv_depth=(64,128,256)):
        super(PointNetfeat, self).__init__()
        self.conv_depth = conv_depth
        self.conv1 = torch.nn.Conv1d(3, conv_depth[0], 1)
        self.conv2 = torch.nn.Conv1d(conv_depth[0], conv_depth[1], 1)
        self.conv3 = torch.nn.Conv1d(conv_depth[1], conv_depth[2], 1)
#        self.bn1 = nn.BatchNorm1d(conv_depth[0])
#        self.bn2 = nn.BatchNorm1d(conv_depth[1])
#        self.bn3 = nn.BatchNorm1d(conv_depth[2])

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.conv1(x))

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.conv_depth[2])
        return x
        
class SimuNetWithSurface(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256), dropout=False):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(SimuNetWithSurface, self).__init__()

        self.pc_layers = PointNetfeat()
        
        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, conv_depths[0], conv_depths[0], dropout=dropout))
        encoder_layers.extend([Encoder3D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1], dropout=dropout)
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i], dropout=dropout)
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last3D(conv_depths[1], conv_depths[0], out_channels, dropout=dropout))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(conv_depths[-2]+256+10, conv_depths[-1]+256, conv_depths[-1]+128, conv_depths[-2], dropout=dropout)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, pc, kinematics):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        pc_feat = self.pc_layers(pc)
        features = torch.cat((pc_feat, kinematics), axis=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        features = features.repeat(1, 1, x_enc[-1].size()[2], x_enc[-1].size()[3], x_enc[-1].size()[4])
        
        x_centre = torch.cat((x_enc[-1], features), axis=1)

        x_dec = [self.center(x_centre)]
        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            x_opposite = x_enc[-1-dec_layer_idx]
            x_cat = torch.cat(
                [pad_to_shape(x_dec[-1], x_opposite.shape), x_opposite],
                dim=1
            )
            x_dec.append(dec_layer(x_cat))

        return x_dec[-1]


class SimuNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depth= (256,256,256,512,512), dropout=False):

        super(SimuNet, self).__init__()
        self.pc_layers = PointNetfeat()
        
        layers = [
            nn.Conv3d(in_channels, conv_depth[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(conv_depth[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(conv_depth[0], conv_depth[1], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(conv_depth[1], conv_depth[2], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[2]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(conv_depth[2], conv_depth[3], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[3]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(conv_depth[3], conv_depth[4], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[4]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(conv_depth[4], out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x, pc, kinematics):
        pc_feat = self.pc_layers(pc)
        features = torch.cat((pc_feat, kinematics), axis=1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        features = features.repeat(1, 1, x.size()[2], x.size()[3], x.size()[4])
        x_pc = torch.cat((x, features), axis=1)
        
        return self.layers(x_pc)
        
# modified from https://github.com/cosmic-cortex/pytorch-UNet
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depths=(64, 128, 256), dropout=False):#, 512, 1024)):
        assert len(conv_depths) > 2, 'conv_depths must have at least 3 members'

        super(UNet3D, self).__init__()

        # defining encoder layers
        encoder_layers = []
        encoder_layers.append(First3D(in_channels, conv_depths[0], conv_depths[0], dropout=dropout))
        encoder_layers.extend([Encoder3D(conv_depths[i], conv_depths[i + 1], conv_depths[i + 1], dropout=dropout)
                               for i in range(len(conv_depths)-2)])

        # defining decoder layers
        decoder_layers = []
        decoder_layers.extend([Decoder3D(2 * conv_depths[i + 1], 2 * conv_depths[i], 2 * conv_depths[i], conv_depths[i], dropout=dropout)
                               for i in reversed(range(len(conv_depths)-2))])
        decoder_layers.append(Last3D(conv_depths[1], conv_depths[0], out_channels, dropout=dropout))

        # encoder, center and decoder layers
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.center = Center3D(conv_depths[-2], conv_depths[-1], conv_depths[-1], conv_depths[-2], dropout=dropout)
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x, return_all=False):
        x_enc = [x]
        for enc_layer in self.encoder_layers:
            x_enc.append(enc_layer(x_enc[-1]))

        x_dec = [self.center(x_enc[-1])]
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
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
        ]

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
#            nn.MaxPool3d(kernel_size=downsample_kernel),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
        ]
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center3D, self).__init__()

        layers = [
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)

        ]
        
        self.center = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.center(x)
        return x 


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(Last3D, self).__init__()

        layers = [
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1),
#            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=dropout),
            nn.Conv3d(middle_channels, out_channels, kernel_size=1),
#            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Tanh()
        ]

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)

#################2D model#############################
#adapted from https://github.com/milesial/Pytorch-UNet
# sub-parts of the U-Net model

    
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

    
# full assembly of the sub-parts to form the complete net

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2D, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
#        self.down3 = down(256, 256)
        self.up2 = up(512, 128)
        self.up3 = up(391, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, out_channels)

    def forward(self, kinematics, mesh):
        x1 = self.inc(mesh)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
#        x4 = self.down3(x3)
        
        kinematics = kinematics.view(kinematics.size()[0], kinematics.size()[1],1,1)
        kinematics = kinematics.repeat(1,1,x3.size()[2],x3.size()[3])
        x3 = torch.cat((x3, kinematics), axis=1)
        
#        x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

