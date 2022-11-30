from re import sub
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
from det3d.models.losses.centernet_loss import FastFocalLoss
from det3d.core.input.voxel_generator import VoxelGenerator
from torch.nn.utils.rnn import pad_sequence
from .. import builder
import logging
import torch 
from copy import deepcopy 
import numpy as np
import spconv

import torch.nn.functional as F
from det3d.models.utils import Empty, GroupNorm, Sequential
from torch.nn.modules.conv import Conv2d
from torch import nn


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
           
    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            neck  = self.neck(x)

        return neck, voxel_feature, x

    def forward(self, example, return_loss=True, return_feature=False, return_recon_feature = False, **kwargs):
        

        if 'dense_voxels' in example:
            voxels = example["dense_voxels"]
            coordinates = example["dense_coordinates"]
            num_points_in_voxel = example["dense_num_points"]
            num_voxels = example["dense_num_voxels"]
        else:
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, _, F_D_a = self.extract_feat(data)

        if return_recon_feature:

            voxels = example["reconstruction_voxels"]
            coordinates = example["reconstruction_coordinates"]
            num_points_in_voxel = example["reconstruction_num_points"]
            num_voxels = example["reconstruction_num_voxels"]
            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
            )
            input_features = self.reader(data["features"], data["num_voxels"])
            F_D_b, _ = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        preds = self.bbox_head(x)

        if return_loss:
            if return_feature == False:
                return self.bbox_head.loss(example, preds)
            else:
                return self.bbox_head.loss(example, preds), F_D_a, F_D_b
        else:
            if return_feature == False:
                return self.bbox_head.predict(example, preds, self.test_cfg)
            else:
                if return_recon_feature:
                    return preds, F_D_a, F_D_b
                else:
                    return  self.bbox_head.predict(example, preds, self.test_cfg), F_D_a, F_D_b

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, voxel_feature,F_D_a = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None,F_D_a, F_D_a


@DETECTORS.register_module
class KD_VoxelNet(VoxelNet):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
     
    def extract_feat(self, data, train_pcm = True):
        input_features = self.reader(data["features"], data["num_voxels"])
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )

        if self.with_neck:
            x, gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4, F_S_a, F_S_b = self.neck(x)

        return x, gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4, F_S_a, F_S_b, voxel_feature

    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):

        gt_mask = gt.sum(1) != 0
        count_pos = gt_mask.sum()
        count_neg = (~gt_mask).sum()
        beta = count_neg/count_pos
        loss = F.binary_cross_entropy_with_logits(gen_mask[:,0],gt_mask.float(),pos_weight= beta) 

        grid = grid * gt_mask[:,None]
        gt = gt[:,:3] - grid
        gt_ind = gt != 0
        
        com_loss = F.l1_loss(gen_offset[gt_ind], gt[gt_ind])

        return loss, com_loss


    def forward(self, example, return_loss=True, return_feature=False, **kwargs):

        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        if return_loss:
            reconstruction_voxels = example["reconstruction_voxels_2"]
            reconstruction_coordinates = example["reconstruction_coordinates_2"]
            reconstruction_num_voxels = example["reconstruction_num_voxels_2"]
            reconstruction_num_points_in_voxel = example['reconstruction_num_points_2']
            sparse_shape = np.array(example["shape"][0][::-1]/2).astype('int64')

            coors = reconstruction_coordinates.int()
            input_feature = self.reader(reconstruction_voxels,reconstruction_num_points_in_voxel)
            reconstruction_gt = spconv.SparseConvTensor(input_feature, coors, sparse_shape, len(reconstruction_num_voxels))
            reconstruction_gt = reconstruction_gt.dense()

            reconstruction_voxels = example["reconstruction_voxels_4"]
            reconstruction_coordinates = example["reconstruction_coordinates_4"]
            reconstruction_num_voxels = example["reconstruction_num_voxels_4"]
            reconstruction_num_points_in_voxel = example['reconstruction_num_points_4']
            sparse_shape = np.array(example["shape"][0][::-1]/4).astype('int64')

            coors = reconstruction_coordinates.int()
            input_feature = self.reader(reconstruction_voxels,reconstruction_num_points_in_voxel)
            reconstruction_gt_4 = spconv.SparseConvTensor(input_feature, coors, sparse_shape, len(reconstruction_num_voxels))
            reconstruction_gt_4 = reconstruction_gt_4.dense()

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, gen_offset_2, gen_mask_2, gen_offset_4, gen_mask_4, F_S_a, F_S_b, voxel_feature = self.extract_feat(data)
        

        if self.training :
            grid_4, grid = None, None
            N,_,D,H,W = gen_offset_2.shape
            zs, ys, xs = torch.meshgrid([torch.arange(0,D),torch.arange(0, H), torch.arange(0, W)])
            ys = ys * (150.4/H) - 75.2 + (150.4/H)/2
            xs = xs * (150.4/W) - 75.2 + (150.4/H)/2
            zs = zs * (6/D) - 2 + (6/D)/2
            grid = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(gen_offset_2)
            if return_loss:
                N,_,D,H,W = reconstruction_gt_4.shape
                zs, ys, xs = torch.meshgrid([torch.arange(0,D),torch.arange(0, H), torch.arange(0, W)])
                ys = ys * (150.4/H) - 75.2 + (150.4/H)/2
                xs = xs * (150.4/W) - 75.2 + (150.4/H)/2
                zs = zs * (6/D) - 2 + (6/D)/2
                grid_4 = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(reconstruction_gt_4)
                mask_loss_4, offset_loss_4 = self.mask_offset_loss(gen_offset_4,gen_mask_4, reconstruction_gt_4, grid_4)

                mask_loss_2, offset_loss_2 = self.mask_offset_loss(gen_offset_2, gen_mask_2, reconstruction_gt, grid)
                mask_loss = mask_loss_2 + mask_loss_4
                comp_loss = offset_loss_2 + offset_loss_4
        else:
            mask_loss_4, offset_loss_4, mask_loss_2, offset_loss_2 = 0, 0, 0, 0
        
        preds = self.bbox_head(x) 


        if return_loss:
            if return_feature == False:
                return self.bbox_head.loss(example, preds), preds
            else:
                return self.bbox_head.loss(example, preds), F_S_a, F_S_b, preds ,mask_loss, comp_loss
        else:
            if return_feature == False:
                return self.bbox_head.predict(example, preds, self.test_cfg)
            else:
                return self.bbox_head.predict(example, preds, self.test_cfg), F_S_a, F_S_b
    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, _, _, _, _, F_S_a, F_S_b, voxel_feature= self.extract_feat(data, train_pcm=False)
        
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, voxel_feature, None, F_S_a, F_S_b