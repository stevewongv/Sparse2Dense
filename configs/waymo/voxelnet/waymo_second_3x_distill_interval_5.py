import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor


tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[2.08, 4.73, 1.77],
            anchor_ranges=[-74.88, -74.88, 0, 74.88, 74.88, 0],
            rotations=[0, 1.57],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="VEHICLE",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.84, 0.91, 1.74],
            anchor_ranges=[-74.88, -74.88, 0, 74.88, 74.88, 0],
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="PEDESTRIAN",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.84, 1.81, 1.77],
            anchor_ranges=[-74.88, -74.88, 0, 74.88, 74.88, 0],
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.3,
            class_name="CYCLIST",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,
)

model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleFHD", num_input_features=5, ds_factor=8,
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5,],
        ds_layer_strides=[1,],
        ds_num_filters=[128,],
        us_layer_strides=[1,],
        us_num_filters=[128,],
        num_input_features=128,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128,]),
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=2.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=2.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=0.2,
        ),
        direction_offset=0.0,
    ),
)

# model settings
S_model = dict(
    type="KD_VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleFHD", num_input_features=5, ds_factor=8,
    ),
    neck=dict(
        type="S2D_RPN",
        layer_nums=[5,],
        ds_layer_strides=[1,],
        ds_num_filters=[128,],
        us_layer_strides=[1,],
        us_num_filters=[128,],
        num_input_features=128,
        logger=logging.getLogger("S2D_RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128,]),
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=2.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=2.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=0.2,
        ),
        direction_offset=0.0,
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(S_model),
    debug=False,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    max_per_img=4096,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.01,
    ),
    score_threshold=0.1,
    pc_range=[-74.88, -74.88],
    out_size_factor=get_downsample_factor(S_model),
)

# dataset settings
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "../data/waymo"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="../data/waymo/dbinfos_train_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
        dict(CYCLIST=10),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                VEHICLE=5,
                PEDESTRIAN=5,
                CYCLIST=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) 
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.785, 0.785],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
    distillation= True
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
    distillation= True
)


voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=150000,
    distillation=True
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat", distillation=True),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat",),
]

train_anno = "../data/waymo/infos_train_01sweeps_filter_zero_gt.pkl"
val_anno = "../data/waymo/infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        load_interval=5,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.3,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = './work_dirs/waymo_second_3x_dense_interval_5/epoch_30.pth'
resume_from = None
workflow = [("train", 1)]
distillation = True