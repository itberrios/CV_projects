import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MobileNetV3Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
    
    def forward(self, x):
        """ Passes input theough MobileNetV3 backbone feature extraction layers
            layers to add connections to
                - 1:  1/4 res
                - 3:  1/8 res
                - 7, 8:  1/16 res
                - 10, 11: 1/32 res
           """
        skips = nn.ParameterDict()
        for i in range(len(self.backbone) - 1):
            x = self.backbone[i](x)
            # add skip connection outputs
            if i in [1, 3, 7, 8, 10, 11]:
                skips.update({f"l{i}_out" : x})

        return skips
    

# Chained Residual Pooling
class CRPBlock(nn.Module):
    def __init__(self, in_chans, out_chans, n_stages=4, groups=False):
        super().__init__()

        self.n_stages = n_stages
        groups = in_chans if groups else 1
        self.mini_blocks = nn.ModuleList()
        for _ in range(n_stages):
            self.mini_blocks.append(nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False, groups=groups))
            self.mini_blocks.append(nn.MaxPool2d(kernel_size=5, stride=1, padding=2))

    
    def forward(self, x):
        out = x
        for block in self.mini_blocks:
            out = block(out)
            x = x + out

        return x


class LightWeightRefineNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 1x1 convs to convert encoder channels to 256
        self.conv1 = nn.Conv2d(96, 256, kernel_size=1, stride=1, bias=False) # 1/32 res
        self.conv2 = nn.Conv2d(96, 256, kernel_size=1, stride=1, bias=False) # 1/32 res
        self.conv3 = nn.Conv2d(48, 256, kernel_size=1, stride=1, bias=False) # 1/16 res
        self.conv4 = nn.Conv2d(48, 256, kernel_size=1, stride=1, bias=False) # 1/16 res
        self.conv5 = nn.Conv2d(24, 256, kernel_size=1, stride=1, bias=False) # 1/8 res
        self.conv6 = nn.Conv2d(16, 256, kernel_size=1, stride=1, bias=False) # 1/4 res

        # CRP Blocks
        self.crp1 = CRPBlock(256, 256, 4, groups=False)
        self.crp2 = CRPBlock(256, 256, 4, groups=False)
        self.crp3 = CRPBlock(256, 256, 4, groups=False)
        self.crp4 = CRPBlock(256, 256, 4, groups=True)

        self.conv_adapt1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.conv_adapt2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.conv_adapt3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)


        # output heads
        self.pre_depth = nn.Conv2d(256, 256, kernel_size=1, groups=256, bias=False)
        self.depth = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=True)

        self.pre_seg = nn.Conv2d(256, 256, kernel_size=1, groups=256, bias=False)
        self.seg = nn.Conv2d(256, num_classes, kernel_size=3, padding=1, bias=True)

        self.relu6 = nn.ReLU6(inplace=True)


    def forward(self, skips):
        # skips = ['l1_out', 'l3_out', 'l7_out', 'l8_out', 'l10_out', 'l11_out']

        # smallest res CRP skip
        l11 = self.conv1(skips['l11_out'])
        l10 = self.conv2(skips['l10_out'])
        l10 = self.relu6(l10 + l11)
        l10 = self.crp1(l10)
        l10 = self.conv_adapt1(l10)
        l10 = nn.Upsample(size=skips['l8_out'].size()[2:], mode='bilinear', align_corners=False)(l10)

        l8 = self.conv3(skips['l8_out'])
        l7 = self.conv4(skips['l7_out'])
        l7 = self.relu6(l7 + l8 + l10)
        l7 = self.crp2(l7)
        l7 = self.conv_adapt3(l7)
        l7 = nn.Upsample(size=skips['l3_out'].size()[2:], mode='bilinear', align_corners=False)(l7)

        l3 = self.conv5(skips['l3_out'])
        l3 = self.relu6(l7 + l3)
        l3 = self.crp3(l3)
        l3 = self.conv_adapt2(l3)
        l3 = nn.Upsample(size=skips['l1_out'].size()[2:], mode='bilinear', align_corners=False)(l3)

        # largest res CRP skip
        l1 = self.conv6(skips['l1_out'])
        l1 = self.relu6(l1 + l3)
        l1 = self.crp4(l1)

        # pass through output heads
        out_seg = self.pre_seg(l3)
        out_seg = self.relu6(out_seg)
        out_seg = self.seg(out_seg)

        out_depth = self.pre_depth(l3)
        out_depth = self.relu6(out_depth)
        out_depth = self.depth(out_depth)

        return out_seg, out_depth
    

class MultiTaskNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    
    def forward(self, x):
        height, width = x.size()[-2:]

        skips = self.encoder(x)
        out_seg, out_depth = self.decoder(skips)
        
        out_seg = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(out_seg)
        out_depth = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)(out_depth)
        
        return out_seg, out_depth



if __name__ == '__main__':
    from torchvision.models import mobilenet_v3_small
    
    num_seg_classes = 23
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mobilenet_backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')

    encoder = MobileNetV3Backbone(mobilenet_backbone.features)
    decoder = LightWeightRefineNet(num_seg_classes)
    model = MultiTaskNetwork(encoder, decoder).to(device)
    model.eval()

    test_image = torch.rand((batch_size, 3, 256, 512)).float().to(device)
    out_seg, out_depth = model(test_image)

    print(out_seg.size(), out_depth.size())
