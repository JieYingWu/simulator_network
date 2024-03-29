import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/fxia22/pointnet.pytorch
class PointNetfeat(nn.Module):
    def __init__(self, conv_depth=(16,32,64)):
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
    
class SimuAttentionNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv_depth= (32, 64, 128, 128, 128, 256, 256, 512, 512, 1024), dropout=False, layer=[128*3,256*3,512*3,1024*3,2025*3, 2048*6, 1024*6, 2048*6]):

        super(SimuAttentionNet, self).__init__()
#        self.pc_layers = PointNetfeat()
#        self.pc_pos_layers = PointNetfeat(conv_depth=(64,128,325))
        pos_layers = [
            nn.Linear(in_channels*2, layer[0]),
 #           nn.BatchNorm1d(layer[0]),
            nn.ReLU(inplace=True),
            nn.Linear(layer[0], layer[1]),
#            nn.BatchNorm1d(layer[1]),
            nn.ReLU(inplace=True),
            nn.Linear(layer[1], layer[2]),
#            nn.BatchNorm1d(layer[2]),
            nn.ReLU(inplace=True),
#            nn.Linear(layer[2], layer[3]),
#            nn.BatchNorm1d(layer[3]),
#            nn.ReLU(inplace=True),
            nn.Linear(layer[2], layer[4]),
#            nn.BatchNorm1d(layer[4]),
            nn.ReLU(inplace=True)
            ]

        vel_layers = [
            nn.Linear(in_channels, layer[1]),
 #           nn.BatchNorm1d(layer[0]),
            nn.ReLU(inplace=True),
#            nn.Linear(layer[0], layer[1]),
#            nn.BatchNorm1d(layer[1]),
#            nn.ReLU(inplace=True),
            nn.Linear(layer[1], layer[2]),
#            nn.BatchNorm1d(layer[2]),
            nn.ReLU(inplace=True),
#            nn.Linear(layer[2], layer[3]),
#            nn.BatchNorm1d(layer[3]),
#            nn.ReLU(inplace=True),
            nn.Linear(layer[2], layer[4]),
#            nn.BatchNorm1d(layer[4]),
            nn.ReLU(inplace=True)
            ]

        # mesh_layers = [
        #     nn.Conv3d(in_channels, conv_depth[0], kernel_size=3, padding=1),
        #     nn.BatchNorm3d(conv_depth[0]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout3d(p=dropout),
        #     nn.Conv3d(conv_depth[0], conv_depth[1], kernel_size=3, padding=1),
        #     nn.BatchNorm3d(conv_depth[1]),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout3d(p=dropout)
        # ]
        
        layers = [
            nn.Linear(layer[4], layer[5]),
#            nn.BatchNorm3d(conv_depth[5]),
            nn.ReLU(inplace=True),
#            nn.Dropout3d(p=dropout),
            nn.Linear(layer[5], layer[6]),
#            nn.BatchNorm3d(conv_depth[6]),
            nn.ReLU(inplace=True),
#            nn.Dropout3d(p=dropout),
#            nn.Linear(layer[6], layer[7]),
#            nn.BatchNorm3d(conv_depth[7]),
#            nn.Dropout3d(p=dropout),
#            nn.ReLU(inplace=True),
#            nn.Conv3d(conv_depth[7], conv_depth[8], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[8]),
#            nn.Dropout3d(p=dropout),
#            nn.ReLU(inplace=True),
#            nn.Conv3d(conv_depth[8], conv_depth[9], kernel_size=3, padding=1),
#            nn.BatchNorm3d(conv_depth[9]),
#            nn.Dropout3d(p=dropout),
#            nn.ReLU(inplace=True),
            # nn.Conv3d(conv_depth[9], conv_depth[10], kernel_size=3, padding=1),
            # nn.BatchNorm3d(conv_depth[10]),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(p=dropout),
            # nn.Conv3d(conv_depth[10], conv_depth[11], kernel_size=3, padding=1),
            # nn.BatchNorm3d(conv_depth[11]),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(p=dropout),
            nn.Linear(layer[6], layer[4]),
            nn.Tanh()
        ]

        self.pos_layers = nn.Sequential(*pos_layers)
        self.vel_layers = nn.Sequential(*vel_layers)
 #       self.mesh_layers = nn.Sequential(*mesh_layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, kinematics):#, pc):
#        pc_features = self.pc_layers(pc)
#        pc_features = pc_features.unsqueeze(2).unsqueeze(3).unsqueeze(4)
#        pc_features = pc_features.repeat(1, 1, x.size()[2], x.size()[3], x.size()[4])

#        pc_pos = self.pc_pos_layers(pc)
#        pc_pos = pc_pos.reshape(pc_pos.size()[0], 1, 13, 5 ,5)
#        pc_pos = pc_pos.repeat(1, 64, 1, 1, 1)
        pos = torch.cat((kinematics[:,0:3], kinematics[:,7:10]), axis=1)
#        pos = pos.repeat(1, 1, x.size()[2], x.size()[3], x.size()[4])
        pos_features = self.pos_layers(pos)
        
#        vel = kinematics[:,7:10]
#        vel = vel.repeat(1, 1, x.size()[2], x.size()[3], x.size()[4])        
#        vel_features = self.vel_layers(vel)

#        mesh_features = self.mesh_layers(x)
#        x = x.reshape(x.size()[0], -1)
#        robot_features = pos_features.reshape(pos_features.size()[0], 3, 25, 9, 9) #* vel_features
#        pc_features = pc_pos * pc_features
#        features = torch.cat((robot_features, pc_features), axis=1)
        x = pos_features #* vel_features
#        x = torch.cat((x, pos_features*vel_features), axis=1)
#        x = self.layers(x)
        x = x.reshape(x.size()[0], 3, 25, 9, 9)
        return x

        
# modified from https://github.com/cosmic-cortex/pytorch-UNet
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.inc = inconv3D(in_channels, 64)
        self.down1 = down3D(64, 128)
        self.down2 = down3D(128, 256)
        self.kinematics_layers = kinematics_layers(6, 72)
#        self.down3 = down3D(256, 256)
#        self.up2 = up3D(522, 128)
        self.up3 = upSpecial(384, 64)
        self.up4 = up3D(128, 64)
        self.outc = outconv3D(64, out_channels)

    def forward(self, mesh, kinematics):
        x1 = self.inc(mesh)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
#        x4 = self.down3(x3)
        
        k = self.kinematics_layers(kinematics)
        k = k.reshape(k.size()[0],3,6,2,2)
        x3 = torch.cat((x3, k), axis=1)

#kinematics should be 6144 large
        
#        x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class kinematics_layers(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kinematics_layers, self).__init__()
        self.layer = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        x = self.layer(x)
        return x

    
    
class double_conv3D(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
#            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
#            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv3D, self).__init__()
        self.conv = double_conv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down3D, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv3D(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up3D(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=False):
        super(up3D, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2,
                        diffZ // 2, diffZ - diffZ//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    
class upSpecial(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=False):
        super(upSpecial, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(259, 256, 2, stride=2)

        self.conv = double_conv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2,
                        diffZ // 2, diffZ - diffZ//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    
class outconv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3D, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


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

