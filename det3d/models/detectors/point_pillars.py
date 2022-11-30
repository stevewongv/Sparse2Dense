from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 

import torch.nn.functional as F
import torch
import numpy as np
import spconv

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
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
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x_fea = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x_fea)
        return x,x_fea

    def forward(self, example, return_loss=True, **kwargs):

        if "dense_voxels" in example:
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

        x, F_D_a = self.extract_feat(data)

        if not return_loss:

           
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
            input_features = self.reader(
                data["features"], data["num_voxels"], data["coors"]
            )
            F_D_b = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )


        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return preds,  F_D_a, F_D_b

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

        x,F_D_a = self.extract_feat(data)
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
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None, None, F_D_a, F_D_a

@DETECTORS.register_module
class KD_PointPillars(PointPillars):
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
        super(KD_PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        F_S_a, F_S_b, gen_offset, gen_mask = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(F_S_a)
        return x, F_S_a, F_S_b, gen_offset, gen_mask
    
    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):

        # grid 
        gt_mask = gt.sum(1) != 0
        count_pos = gt_mask.sum()
        count_neg = (~gt_mask).sum()
        beta = count_neg/count_pos
        loss = F.binary_cross_entropy_with_logits(gen_mask[:,0],gt_mask.float(),pos_weight= beta)

        grid = grid * gt_mask[:,None]
        gt = gt[:,:3] - grid
        gt_ind = gt != 0
        
        com_loss = F.l1_loss(gen_offset[gt_ind],gt[gt_ind])
        
        return loss, com_loss

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        if return_loss:
            reconstruction_voxels = example["reconstruction_voxels"]
            reconstruction_coordinates = example["reconstruction_coordinates"]
            reconstruction_num_voxels = example["reconstruction_num_voxels"]
            reconstruction_num_points_in_voxel = example['reconstruction_num_points']
            sparse_shape = np.array(example["shape"][0][::-1]).astype('int64')
            coors = reconstruction_coordinates.int()
            input_feature = (reconstruction_voxels[:, :, : 5].sum( dim=1, keepdim=False
                    ) /reconstruction_num_points_in_voxel.type_as(reconstruction_voxels).view(-1, 1)).contiguous()
            reconstruction_gt = spconv.SparseConvTensor(input_feature, coors, sparse_shape, len(reconstruction_num_voxels))
            reconstruction_gt = reconstruction_gt.dense() # 4,5,1,486,868


        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x, F_S_a, F_S_b ,gen_offset, gen_mask= self.extract_feat(data)
        preds = self.bbox_head(x)

        N,_,D,H,W = gen_offset.shape
        zs, ys, xs = torch.meshgrid([torch.arange(0,D),torch.arange(0, H), torch.arange(0, W)])
        ys = ys * (150.4/H) - 75.2 + (150.4/H)/2
        xs = xs * (150.4/W) - 75.2 + (150.4/H)/2
        zs = zs * (6/D) - 2 + (6/D)/2
        grid = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(gen_offset)

        if return_loss:
            mask_loss, offset_loss = self.mask_offset_loss(gen_offset, gen_mask, reconstruction_gt, grid)

        if return_loss:
            return self.bbox_head.loss(example, preds), F_S_a, F_S_b ,preds,mask_loss, offset_loss
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
        
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

        x, F_S_a, F_S_b ,_, _ = self.extract_feat(data)
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
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None, None, F_S_a, F_S_b

