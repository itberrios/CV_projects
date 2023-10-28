import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# add RAFT to core path
sys.path.append('RAFT/core')

from RAFT.core.extractor import BottleneckBlock
from RAFT.core.corr import CorrBlock
from RAFT.core.utils.utils import coords_grid


class Network(nn.Module):
    def __init__(self, fnet, corr_radius=4, freeze_encoder=True, p=0.5, device='cuda'):
        super().__init__()

        self.fnet = fnet
        self.corr_radius = corr_radius
        self.device = device

        # conv layers
        self.bottle1 = BottleneckBlock(in_planes=324, planes=32, norm_fn='group', stride=2).to(self.device)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, dilation=1, padding=0).to(self.device)
        # self.conv1 = nn.Conv2d(in_channels=324, out_channels=64, kernel_size=3, stride=2, dilation=1, padding=0).to(self.device)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, dilation=1, padding=0).to(self.device)

        # fully connected layers
        self.fc1 = nn.Linear(in_features=352, out_features=1).to(self.device)

        # self.norm_1 = nn.GroupNorm(num_groups=2, num_channels=64)
        # self.norm_1 = nn.GroupNorm(num_groups=2, num_channels=1)

        # self.dropout2d_1 = nn.Dropout2d(p=p)
        # self.dropout2d_2 = nn.Dropout2d(p=p)
        # self.dropout1d = nn.Dropout1d(p=p)

        # freeze encoder weights
        if freeze_encoder:
            for child in self.fnet.children():
                for param in child.parameters():
                    param.requires_grad = False


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    

    def forward(self, image1, image2):
        # prep images
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with torch.autocast(device_type=self.device, enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # get correlation function (and correlation pyramid)
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
        
        _, coords1 = self.initialize_flow(image1)

        # index correlation volume to get correlation features
        corr_features = corr_fn(coords1.detach())

        # TEMP
        # print(corr_features.shape)
        # print(self.bottle1(corr_features).shape) # (b, 32, 34, 45)

        # reduce to extract speed estimation
        # out = F.relu(self.norm_1(self.conv1(corr_features)))

        out = self.bottle1(corr_features)
        out = F.relu(self.conv1(out))
        # out = self.dropout2d_1(out)
        # out = F.relu(self.conv2(out))
        # out = self.dropout2d_2(out)

        # print(out.shape)

        out = out.reshape(out.size()[0], -1)

        # print(out.shape)

        out = self.fc1(out)
        # out = self.dropout1d(out)

        return out.squeeze()
    

# example
if __name__ == '__main__':

    raft_model = load_model("RAFT/models/raft-sintel.pth", args=Args())

    fnet = raft_model.module.fnet
    # cnet = raft_model.module.cnet

    model = Network(fnet)
    model.eval();
