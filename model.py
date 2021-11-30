import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, vgg16, vgg19, inception_v3, densenet121, densenet161

class SELayer(nn.Module):
    def __init__(self, channel=3, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

# ------------------------ Segmentation Model -------------------------------- #

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels = 3, **kwargs):
        super().__init__()
        
        nb_filter = [32,64,128,256,512]
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output
        
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        
        nb_filter = [32,64,128,256,512]
        
        self.deep_supervision = deep_supervision
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1= VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size = 1)
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final1(x0_2)
            output3 = self.final1(x0_3)
            output4 = self.final1(x0_4)
            return [output1, output2, output3, output4]
        
        else:
            output = self.final(x0_4)
            return output

#---------ResUNet----------

class ResNetEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        architecture = resnet34()
        self.conv1 = architecture.conv1
        self.bn1 = architecture.bn1
        self.relu = architecture.relu
        self.maxpool = architecture.maxpool
        self.layer1 =architecture.layer1
        self.layer2 = architecture.layer2
        self.layer3 = architecture.layer3
        self.layer4 = architecture.layer4
        del architecture.avgpool
        del architecture.fc

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(self.layer1(x1))
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1, x2, x3, x4, x5]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch+skip_ch, out_channels=out_ch, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features = out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features = out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SupervisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.dense = nn.Linear(128*8*8, 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

class ResUnet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, deep_supervision=False):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.classes = num_classes
        self.decoder1 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        self.decoder4 = DecoderBlock(64, 64, 32)
        self.decoder5 = DecoderBlock(32, 0, 16)
        self.seg_head = nn.Conv2d(in_channels = 16, out_channels=self.classes, kernel_size=(3,3), padding=3//2)
        self.deep_supervision = deep_supervision
        self.supervision_block = SupervisionBlock()


    def forward(self, x):
        enc_outs = self.encoder(x)
        supervision_out = self.supervision_block(enc_outs[-1])

        x1 = self.decoder1(enc_outs[-1], enc_outs[-2])
        x2 = self.decoder2(x1, enc_outs[-3])
        x3 = self.decoder3(x2, enc_outs[-4])
        x4 = self.decoder4(x3, enc_outs[-5])
        x5 = self.decoder5(x4, skip=None)
        out = self.seg_head(x5)
        if self.deep_supervision:
            return out, supervision_out
        return out