"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from det3d.models.utils import get_paddings_indicator
from torch import nn
from torch.nn import functional as F
from ..registry import BACKBONES, READERS
from ..utils import build_norm_layer
from det3d.models.utils import  Sequential


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@READERS.register_module
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype

        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@BACKBONES.register_module
class PointPillarsScatter(nn.Module):
    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):

        self.nx = input_shape[0]
        self.ny = input_shape[1]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        





        return batch_canvas

@BACKBONES.register_module
class PointPillarsScatter_S2D(nn.Module):
    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

        # S2D module for PointPillar
        self.encoder_1 = Sequential(     #  N,64,468,468
            nn.MaxPool2d(2,2),           #  N,64,234,234
            nn.Conv2d(64,32,1,1,0),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,32,2,2),        #  N,64,117,117
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.GELU(),                   # 2,64,117,117
             
        )
        self.encoder_2 = Sequential(
            nn.Conv2d(128,128,3,2,1),    #  N,64,59,59
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.convnext_block_1 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,59,59], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.convnext_block_2 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,59,59], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        
        self.convnext_block_3 = Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.LayerNorm([256,59,59], eps=1e-6),
            nn.Conv2d(256,256*4,1,1,0),
            nn.GELU(),
            nn.Conv2d(256*4,256,1,1,0),
        )
        self.decoder_1 = Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample((117,117))              # 2,64,117,117
        )

        self.decoder_2 = Sequential(
            nn.Conv2d(128+128,64,3,1,1),        # 2,64,117,117
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64,64,4,2,1),    #  N,64,234,234
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,64,1,1,0),             #  N,64,234,234
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2)         #  N,64,468,468
        )

        self.fusion_sparse = Sequential(
            nn.Conv2d(64,64,1,1,0),
            nn.BatchNorm2d(num_input_features),
            nn.GELU(),
        )

        self.fusion_dense = Sequential(
            nn.Conv2d(64,64,1,1,0),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # PCR module for PointPillar

        self.generator = Sequential( 
            nn.Conv3d(64,32,1,1,0), 
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32,16,1,1,0),
            nn.BatchNorm3d(16),
            nn.GELU(),
            
        )

        self.gen_out= Sequential(
            nn.Conv3d(16,3,1,1,0)
        )

        self.gen_mask = Sequential(
            nn.Conv3d(16,8,1,1,0),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8,1,1,1,0)
        )
        

        
    def forward(self, voxel_features, coords, batch_size, input_shape):

        self.nx = input_shape[0]
        self.ny = input_shape[1]

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        
        y_1 = self.encoder_1(batch_canvas)                                 # 117
        y_2 = self.encoder_2(y_1)                                          # 59
        att = self.convnext_block_1(y_2) + y_2                             # 59
        att = self.convnext_block_2(att) + att                             # 59
        att = self.convnext_block_3(att) + att                             # 59
        y_3 = torch.cat([self.decoder_1(att) , y_1],1)                     # 117
        F_S_b = self.decoder_2(y_3)                                          # 468
        F_S_a = self.fusion_dense(F_S_b) + self.fusion_sparse(batch_canvas)  # 468
        if self.training:
            N, C, H, W = batch_canvas.shape
            gen = F_S_b.view(N,C,1,H,W)
            gen = self.generator(gen)
            gen_mask = self.gen_mask(gen)
            gen_offset = self.gen_out(gen)
        else:
            gen_mask = None
            gen_offset = None
        

        return F_S_a, F_S_b, gen_offset, gen_mask
