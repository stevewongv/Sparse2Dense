# modified from the single_inference.py by @muzi2045
# from spconv.utils import VoxelGenerator as VoxelGenerator
from itertools import count
from det3d.core import anchor
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.datasets.pipelines.loading import read_single_waymo
from det3d.datasets.pipelines.loading import get_obj
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.torchie import Config
from tqdm import tqdm 
import numpy as np
import pickle 
import open3d as o3d
import argparse
import torch
import time 
import os 
import re

voxel_generator = None 
model = None 
device = None 

def initialize_model(args):
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.S_model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num
    )
    return model 

def voxelization(points, voxel_generator):
    voxel_output = voxel_generator.generate(points)  
    voxels, coords, num_points = \
        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

    return voxels, coords, num_points  

def _process_inputs(points, fp16):
    voxels, coords, num_points = voxel_generator.generate(points)
    with open('./anchors.pkl','rb') as f:
        anchors = pickle.load(f)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)
    anchors = torch.tensor(anchors[None,None], dtype=torch.float32, device=device)

    if fp16:
        voxels = voxels.half()

    inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size],
            anchors = anchors
        )

    return inputs 

def run_model(points, fp16=False):
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        torch.cuda.synchronize()
        start = time.time()
        outputs, x_fea, comp_fea = model(data_dict, return_loss=False, return_feature= True)
        # outputs = model(data_dict, return_loss=False, return_feature= False)
        torch.cuda.synchronize()
        end = time.time()

    return {'boxes': outputs[0]['box3d_lidar'].cpu().numpy(),
        'scores': outputs[0]['scores'].cpu().numpy(),
        'classes': outputs[0]['label_preds'].cpu().numpy(),
        'time':end-start,}
        # 'x_fea':x_fea.cpu().numpy(),
        # 'comp_fea': comp_fea.cpu().numpy()}

def process_example(points, fp16=False):
    output = run_model(points, fp16)

    # assert len(output) == 6
    # assert set(output.keys()) == set(('boxes', 'scores', 'classes', 'time','x_fea','comp_fea'))
    num_objs = output['boxes'].shape[0]
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    return output    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'_(\d+).pkl', text) ]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default=None, type=str
    )
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    print("Please prepare your point cloud in waymo format and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    # Run any user-specified initialization code for their submission.
    model = initialize_model(args)

    latencies = []
    visual_dicts = []
    pred_dicts = {}
    counter = 0 
    times = 0
    lists = list(os.listdir(args.input_data_dir))
    lists.sort(key=natural_keys)
    for frame_name in tqdm(lists):
        if counter == args.num_frame:
            break
        else:
            counter += 1 

        pc_name = os.path.join(args.input_data_dir, frame_name)
        points = read_single_waymo(get_obj(pc_name))
        anno = get_obj('../data/Waymo/val/annos/'+frame_name)
        gt_boxes = np.array([ o['box'] for o in anno['objects'] ])
        gt_num_points =  np.array([ o['num_points'] for o in anno['objects'] ])
        indx = gt_num_points.nonzero()
        gt_class = np.array([ o['label'] for o in anno['objects'] ])
        try:
            gt_boxes = gt_boxes[:,[0,1,2,3,4,5,-1]][indx]
            
            gt_boxes[:,-1] = -gt_boxes[:,-1] 
            gt_class = np.ones([gt_boxes.shape[0]],dtype='int')*4
            gt_score = np.ones([gt_boxes.shape[0]])
        except:
            gt_boxes = []
            gt_class = []
            gt_score = []
       
        detections = process_example(points, args.fp16)
        if len(gt_boxes) != 0:
            detections['boxes'] = np.concatenate([detections['boxes'], gt_boxes],0)
            detections['scores'] = np.concatenate([detections['scores'],gt_score], 0)
            detections['classes'] = np.concatenate([detections['classes'],gt_class],0)
        else:
            detections['boxes'] = None
            detections['scores'] = None
            detections['classes'] = None
        detections['name'] = frame_name
        if counter > 5:
            times += detections['time']

        if args.visual and args.online:
            pcd = o3d.geometry.PointCloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            visual += plot_boxes(detections, args.threshold)

            o3d.visualization.draw_geometries(visual)
        elif args.visual:
            visual_dicts.append({'points': points, 'detections': detections})

        pred_dicts.update({frame_name: detections})

    if args.visual:
        with open(os.path.join(args.output_dir, 'visualization.pkl'), 'wb') as f:
            pickle.dump(visual_dicts, f)

    with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
        pickle.dump(pred_dicts, f)
    print("time: {}".format(1/(times/(len(os.listdir(args.input_data_dir))))))
