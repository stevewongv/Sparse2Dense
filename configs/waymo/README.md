# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.8.1, CUDA 11.1, and CUDNN 8.2.
- The training is conducted on 4 3090 GPUs. 
- Testing times are measured on one 3090 GPU with batch size 1. 
 
## Waymo 3D Detection 

To train S2D, we have multiple training stages for different detectors:

1.  Train the dense detector by replacing objects in original data with our generated dense objects. 
2. Train sparse detector with S2D/PCR module guided by dense detector.
3. For CenterPoint-based methods, we follow original CenterPoint training strategy to perform "two-stage" training.

For example,
Use [dense](voxelnet/waymo_centerpoint_voxelnet_3x_dense_interval_5.py) config file to train the one-stage VoxelNet-based CenterPoint (dense detector) by using dense objects. Then, use [distill](voxelnet/waymo_centerpoint_voxelnet_3x_distill_interval_5.py) config file to train the one-stage VoxelNet-based CenterPoint (sparse detector). Finally, use [two-stage](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_distill_interval_5.py) config file to train the two-stage VoxelNet-based CenterPoint.

Note that due to the limited computation resourse, we trained all models on 20% Waymo Open Dataset.

## CenterPoint-based Method

### One-stage VoxelNet 

| Model   | Veh_L2 | Ped_L2 | Cyc_L2  | Overall mAPH   | 
|---------|--------|--------|---------|--------|
| [VoxelNet](voxelnet/waymo_centerpoint_voxelnet_3x_interval_5.py) | 63.0 | 63.7 | 65.0 | 61.5 | 
| [VoxelNet+S2D](voxelnet/waymo_centerpoint_voxelnet_3x_distill_interval_5.py) | 66.1 | 67.5 | 68.7 | 65.0 |


### Two-stage VoxelNet

By default, we finetune a pretrained [one stage model](voxelnet/waymo_centerpoint_voxelnet_3x.py) for 30 epochs on 20% Waymo Open Dataset. To save GPU memory, we also freeze the backbone weight.  

#### Waymo Open Dataset

| Model   | Split | Veh_L2 | Ped_L2 | Cyc_L2  | Overall mAPH   |
|------------|----|----|--------|---------|--------|
| [CenterPoint](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_interval_5.py) | Val | 65.5 | 66.3 | 66.3 | 63.78 |
| [CenterPoint+S2D](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_distill_interval_5.py) | Val| 68.2 | 70.1 |  69.3| 66.9 |


#### Waymo Domain Adaption Dataset

| Model   | Split | Veh_L2 mAP | Veh_L2 mAPH |  Ped_L2 mAP | Ped_L2 mAPH   |      
|------------|----|----|----|----|---------|
| [CenterPoint](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_interval_5_da.py) | Val | 48.4 | 47.9 | 21.2 | 19.8 | 
| [CenterPoint+S2D](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_distill_interval_5_da.py) | Val| 51.0 | 50.4 |  26.0| 24.7|


### Two-stage PointPillars 

#### Waymo Open Dataset

| Model   | Veh_L2 | Ped_L2 | Cyc_L2  | Overall mAPH   | 
|---------|--------|--------|---------|--------|
| [CenterPoint-Pillar](pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_3x_distill_interval_5.py) | 64.1 | 61.1 | 59.76 | 57.9 | 
| [CenterPoint-Pillar+S2D](pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_distill_interval_5.py) | 68.1 | 66.4 | 65.3 | 63.1 | 

#### Waymo Domain Adaption Dataset

| Model   | Split | Veh_L2 mAP | Veh_L2 mAPH |  Ped_L2 mAP | Ped_L2 mAPH   |      
|------------|----|----|----|----|---------|
| [CenterPoint-Pillar](pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_interval_5_da.py) | Val | 45.3 | 44.6 | 8.8x | 7.3 | 
| [CenterPoint-Pillar+S2D](pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_distill_interval_5_da.py) | Val| 50.1 | 49.6 |  13.3 | 11.4|

## SECOND

#### Waymo Open Dataset

| Model   | Veh_L2 | Ped_L2 | Cyc_L2  | Overall MAPH   |
|---------|--------|--------|---------|------------|
| [SECOND](voxelnet/waymo_second_3x_interval_5.py) | 59.4 | 48.0 | 55.2 | 49.7 |  
| [SECOND+S2D](voxelnet/waymo_second_3x_distill_interval_5.py) | 63.5 | 51.1 | 57.0 | 52.9 | 

#### Waymo Domain Adaption Dataset

| Model   | Split | Veh_L2 mAP | Veh_L2 mAPH |  Ped_L2 mAP | Ped_L2 mAPH   |      
|------------|----|----|----|----|---------|
| [SECOND](voxelnet/waymo_second_3x_interval_5_da.py) | Val | 42.9 | 41.2 | 9.8 | 8.5 | 
| [SECOND+S2D](voxelnet/waymo_second_3x_distill_interval_5_da.py) | Val| 46.3 | 45.0 |  12.2| 10.7|
