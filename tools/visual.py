from det3d.core.bbox.box_np_ops import center_to_corner_box3d, points_in_rbbox
import open3d as o3d
import argparse
import pickle 
import numpy as np

def label2color(label):
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255], [0,255/255,0]]

    return colors[label]

def corners_to_lines(qs, color=[204/255, 0, 0]):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)
    
    return line_set

def plot_boxes(boxes, score_thresh):
    visuals =[] 
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue 
        # if score == 1:
        #     continue

        box = boxes['boxes'][i:i+1]
        label = boxes['classes'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.4        )
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data_dicts = pickle.load(f)
    count=1
    for data in data_dicts:
        print(count)
        #if count not in [234,311,81]: # 44,23, 58, 87 , 81,135,149,179,208,213,296,298,311,398 | 1,2,81,135,149,179,208,213,296,298,311,398,29,117,178,234,237,260,283,287,300,334,349,362,375
        #    count+=1
        #    continue
        
        points = data['points']
        detections = data['detections']
        boxes = data['detections']['boxes']
        scores = data['detections']['scores']
        #indx = (scores>0.5).nonzero()
        #boxes = boxes[indx]
        index_out_box =  ~np.any(points_in_rbbox(points, boxes),axis=1)
        colors = np.array([[255,196,87]]) / 255 *np.ones_like(points[:, :3])
        colors[index_out_box] = np.array([109,108,124]) / 255
        #print(detections['name'])

        pcd = o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud('./visual{}.pcd'.format(1),pcd)
        visual = [pcd]
        num_dets = detections['scores'].shape[0]
        visual += plot_boxes(detections, args.thresh)
        print(visual)
        count+=1

        o3d.visualization.draw_geometries(visual)
