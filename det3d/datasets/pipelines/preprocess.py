import pickle 
from typing_extensions import Annotated
import numpy as np
import os 



from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler, build_anchor_generator, build_similarity_metric, build_box_coder
from det3d.core.anchor.target_assigner import TargetAssigner
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from det3d.datasets.pipelines.loading import get_obj
from ..registry import PIPELINES

from det3d.builder import (
    build_dbsampler,
    
)

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.distillation = cfg.distillation
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]  
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

            point_out_box =  ~np.any(box_np_ops.points_in_rbbox(points, info['gt_boxes']),axis=1)
            combined_points = [points[point_out_box]]
            reconstruction_points = []
            objects = info['gt_signs']
            if self.distillation:
                for obj,box,type in zip(objects,info['gt_boxes'],info['gt_names']):

                    if os.path.exists('../data/waymo/train/gt/'+obj+'.pkl') and type != 'SIGN':
                        gt_point = get_obj('../data/waymo/train/gt/'+obj+'.pkl')['gt']       
                        if type == "VEHICLE":
                            ind = gt_point[:,1] > 0
                            vind = gt_point[:,1] < 0
                            if ind.sum() > vind.sum():
                                gt_point = gt_point[ind]
                                vgt_ind = gt_point.copy()
                                vgt_ind[:,1] *= -1
                                gt_point = np.concatenate([gt_point,vgt_ind],0)
                            else:
                                gt_point = gt_point[vind]
                                vgt_ind = gt_point.copy()
                                vgt_ind[:,1] *= -1
                                gt_point = np.concatenate([gt_point,vgt_ind],0)
                        gt_point[:,:3] = box_np_ops.rotation_points_single_angle(gt_point[:,:3],(np.pi/2+box[-1]),axis=2)
                        gt_point[:,:3] += box[None][:,:3]
                        
                    else:
                        gt_point = points[box_np_ops.points_in_rbbox(points, box[None])[:,0]]
                    
                    combined_points.append(gt_point)
                    if type != 'SIGN':
                        reconstruction_points.append(gt_point)
                dense_points = np.concatenate(np.array(combined_points),0)
                if len(reconstruction_points) == 0:
                    reconstruction_points = np.zeros([1,5])
                else:
                    reconstruction_points = np.concatenate(np.array(reconstruction_points),0)
                    reconstruction_points = reconstruction_points[np.any(box_np_ops.points_in_rbbox(reconstruction_points,info['gt_boxes']),axis=1)]

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    sampled_recon_points = sampled_dict['recon_points']
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    points = np.concatenate([sampled_points, points], axis=0)
                    if self.distillation:
                        dense_points = np.concatenate([sampled_points, dense_points], axis=0)
                        reconstruction_points = np.concatenate([sampled_recon_points, reconstruction_points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            if self.distillation:
                gt_dict["gt_boxes"], points, dense_points, reconstruction_points = prep.random_flip_both_(gt_dict["gt_boxes"], points, dense_points, reconstruction_points)
                
                gt_dict["gt_boxes"], points, dense_points, reconstruction_points = prep.global_rotation_(
                    gt_dict["gt_boxes"], points, dense_points,  reconstruction_points, rotation=self.global_rotation_noise
                )
                gt_dict["gt_boxes"], points, dense_points, reconstruction_points = prep.global_scaling_v2_(
                    gt_dict["gt_boxes"], points, dense_points,  reconstruction_points, *self.global_scaling_noise
                )
                gt_dict["gt_boxes"], points, dense_points, reconstruction_points = prep.global_translate_(
                    gt_dict["gt_boxes"], points, dense_points,  reconstruction_points, noise_translate_std=self.global_translate_std
                )
            else:
                gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
                
                gt_dict["gt_boxes"], points = prep.global_rotation(
                    gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
                )
                gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                    gt_dict["gt_boxes"], points, *self.global_scaling_noise
                )
                gt_dict["gt_boxes"], points = prep.global_translate(
                    gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
                )
            
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        else:
            if self.distillation:
                point_out_box =  ~np.any(box_np_ops.points_in_rbbox(points, info['gt_boxes']),axis=1)
                combined_points = [points[point_out_box]]
                reconstruction_points = []
                objects = info['gt_signs']
                for obj,box,typ in zip(objects,info['gt_boxes'],info['gt_names']):
                    if os.path.exists('../data/waymo/val/gt/'+obj+'.pkl'):
                        gt_point = get_obj('../data/waymo/val/gt/'+obj+'.pkl')['gt']       
                        if typ == "VEHICLE":
                            ind = gt_point[:,1] > 0
                            vind = gt_point[:,1] < 0
                            if ind.sum() > vind.sum():
                                gt_point = gt_point[ind]
                                vgt_ind = gt_point.copy()
                                vgt_ind[:,1] *= -1
                                gt_point = np.concatenate([gt_point,vgt_ind],0)
                            else:
                                gt_point = gt_point[vind]
                                vgt_ind = gt_point.copy()
                                vgt_ind[:,1] *= -1
                                gt_point = np.concatenate([gt_point,vgt_ind],0)
                        gt_point[:,:3] = box_np_ops.rotation_points_single_angle(gt_point[:,:3],(np.pi/2+box[-1]),axis=2)
                        gt_point[:,:3] += box[None][:,:3]
                        
                    else:
                        gt_point = points[box_np_ops.points_in_rbbox(points, box[None])[:,0]]
                    
                    #combined_points.append(gt_point)
                    if typ != 'SIGN':
                        reconstruction_points.append(gt_point)

                    combined_points.append(gt_point)
                    reconstruction_points.append(gt_point)
                dense_points = np.concatenate(np.array(combined_points),0)
                if len(reconstruction_points) == 0:
                    reconstruction_points = np.zeros([1,5])
                else:
                    reconstruction_points = np.concatenate(np.array(reconstruction_points),0)
                    reconstruction_points = reconstruction_points[np.any(box_np_ops.points_in_rbbox(reconstruction_points,info['gt_boxes']),axis=1)]


        if self.shuffle_points:
            np.random.shuffle(points)
            if self.distillation:
                np.random.shuffle(dense_points)

            

        res["lidar"]["points"] = points
        if self.distillation:
            res["lidar"]["dense_points"] = dense_points
            res["lidar"]["reconstruction_points"] = reconstruction_points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        distillation = kwargs.get("distillation", None)
        if distillation is not None:
            cfg.distillation = distillation
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)
        self.distillation = cfg.distillation
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )
        if self.distillation:
            self.voxel_generator_ = VoxelGenerator(
                voxel_size=[x for x in self.voxel_size] ,
                point_cloud_range=self.range,
                max_num_points=self.max_points_in_voxel,
                max_voxels=self.max_voxel_num[0],
            )
            
            self.voxel_generator_2 = VoxelGenerator(
                voxel_size=[x*2 for x in self.voxel_size] ,
                point_cloud_range=self.range,
                max_num_points=self.max_points_in_voxel,
                max_voxels=self.max_voxel_num[0],
            )
            self.voxel_generator_4 = VoxelGenerator(
                voxel_size=[x*4 for x in self.voxel_size] ,
                point_cloud_range=self.range,
                max_num_points=self.max_points_in_voxel,
                max_voxels=self.max_voxel_num[0],
            )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        if self.distillation:
            voxels, coordinates, num_points = self.voxel_generator.generate(
                res["lidar"]["dense_points"], max_voxels=max_voxels 
            )

            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["dense_voxels"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )
            # down_2
            voxels, coordinates, num_points = self.voxel_generator_.generate(
                res["lidar"]["reconstruction_points"].astype('float32'), max_voxels=max_voxels 
            )

            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["reconstruction_voxels"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )
            
            
            # down_2
            voxels, coordinates, num_points = self.voxel_generator_2.generate(
                res["lidar"]["reconstruction_points"].astype('float32'), max_voxels=max_voxels 
            )

            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["reconstruction_voxels_2"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )
            # down_4
            voxels, coordinates, num_points = self.voxel_generator_4.generate(
                res["lidar"]["reconstruction_points"].astype('float32'), max_voxels=max_voxels 
            )

            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            res["lidar"]["reconstruction_voxels_4"] = dict(
                voxels=voxels,
                coordinates=coordinates,
                num_points=num_points,
                num_voxels=num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )
            

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )        

        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"] 
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info

@PIPELINES.register_module
class AssignTarget(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        target_assigner_config = assigner_cfg.target_assigner
        tasks = target_assigner_config.tasks
        box_coder_cfg = assigner_cfg.box_coder

        anchor_cfg = target_assigner_config.anchor_generators
        anchor_generators = []
        for a_cfg in anchor_cfg:
            anchor_generator = build_anchor_generator(a_cfg)
            anchor_generators.append(anchor_generator)
        similarity_calc = build_similarity_metric(
            target_assigner_config.region_similarity_calculator
        )
        positive_fraction = target_assigner_config.sample_positive_fraction
        if positive_fraction < 0:
            positive_fraction = None
        target_assigners = []
        flag = 0

        box_coder = build_box_coder(box_coder_cfg)

        for task in tasks:
            target_assigner = TargetAssigner(
                box_coder=box_coder,
                anchor_generators=anchor_generators[flag: flag + task.num_class],
                region_similarity_calculator=similarity_calc,
                positive_fraction=positive_fraction,
                sample_size=target_assigner_config.sample_size,
            )
            flag += task.num_class
            target_assigners.append(target_assigner)

        self.target_assigners = target_assigners
        self.out_size_factor = assigner_cfg.out_size_factor
        self.anchor_area_threshold = target_assigner_config.pos_area_threshold

    def __call__(self, res, info):
        class_names_by_task = [t.classes for t in self.target_assigners]
        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]
        feature_map_size = grid_size[:2] // self.out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        anchors_by_task = [
            t.generate_anchors(feature_map_size) for t in self.target_assigners
        ]
        anchor_dicts_by_task = [
            t.generate_anchors_dict(feature_map_size) for t in self.target_assigners
        ]
        reshaped_anchors_by_task = [
            t["anchors"].reshape([-1, t["anchors"].shape[-1]]) for t in anchors_by_task
        ]
        matched_by_task = [t["matched_thresholds"] for t in anchors_by_task]
        unmatched_by_task = [t["unmatched_thresholds"] for t in anchors_by_task]

        bv_anchors_by_task = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, -1]])
            for anchors in reshaped_anchors_by_task
        ]

        anchor_caches_by_task = dict(
            anchors=reshaped_anchors_by_task,
            anchors_bv=bv_anchors_by_task,
            matched_thresholds=matched_by_task,
            unmatched_thresholds=unmatched_by_task,
            anchors_dict=anchor_dicts_by_task,
        )

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m][:, [0, 1, 2, 3, 4, 5, -1]])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

        anchorss = anchor_caches_by_task["anchors"]
        anchors_bvs = anchor_caches_by_task["anchors_bv"]
        anchors_dicts = anchor_caches_by_task["anchors_dict"]

        example = {}
        example["anchors"] = anchorss
        # print(anchorss[0].shape)
        # print(len(anchorss))
        # with open('./anchors.pkl','wb') as f :
        #     pickle.dump(anchorss[0],f)
        # assert 1!=1

        if self.anchor_area_threshold >= 0:
            example["anchors_mask"] = []
            for idx, anchors_bv in enumerate(anchors_bvs):
                anchors_mask = None
                # slow with high resolution. recommend disable this forever.
                coors = coordinates
                dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                    coors, tuple(grid_size[::-1][1:])
                )
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                anchors_area = box_np_ops.fused_get_anchors_area(
                    dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size
                )
                anchors_mask = anchors_area > anchor_area_threshold
                example["anchors_mask"].append(anchors_mask)

        if res["mode"] == "train":
            targets_dicts = []
            for idx, target_assigner in enumerate(self.target_assigners):
                if "anchors_mask" in example:
                    anchors_mask = example["anchors_mask"][idx]
                else:
                    anchors_mask = None
                targets_dict = target_assigner.assign_v2(
                    anchors_dicts[idx],
                    gt_dict["gt_boxes"][idx],
                    anchors_mask,
                    gt_classes=gt_dict["gt_classes"][idx],
                    gt_names=gt_dict["gt_names"][idx],
                )
                targets_dicts.append(targets_dict)

            example.update(
                {
                    "labels": [
                        targets_dict["labels"] for targets_dict in targets_dicts
                    ],
                    "reg_targets": [
                        targets_dict["bbox_targets"] for targets_dict in targets_dicts
                    ],
                    "reg_weights": [
                        targets_dict["bbox_outside_weights"]
                        for targets_dict in targets_dicts
                    ],
                }
            )

        res["lidar"]["targets"] = example

        return res, info