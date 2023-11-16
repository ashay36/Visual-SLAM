import numpy as np 
import argparse

from core.model import VisualSLAM
from core.dataset import KittiDataset
from tqdm import tqdm

    

def parse_argument():
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--path', required=True)
    parser.add_argument('--optimize', action='store_true', help='enable pose graph optimization')
    parser.add_argument('--local_window', default=10, type=int, help='number of frames to run the optimization')
    parser.add_argument('--num_iter', default=100, type=int, help='number of max iterations to run the optimization')
    
    return parser.parse_args()

def main():
    
    args = parse_argument()
    
    # Get data params using the dataloader 
    dataset = KittiDataset(args.path)
    camera_matrix = dataset.intrinsic
    ground_truth_poses = dataset.ground_truth
    num_frames = len(dataset)
    
    model = VisualSLAM(camera_matrix, ground_truth_poses, args)

    pred_poses = []
    
    # Iterate over the frames and update the rotation and translation vectors
    for index in tqdm(range(num_frames)):

        frame, _ , _ = dataset[index]
        model(index, frame)
        if index>2:
            x, y, z = model.cur_t[0], model.cur_t[1], model.cur_t[2]
            pred_poses.append([x[0], y[0], z[0]])
        else:
            x, y, z = 0.,  0., 0.
            pred_poses.append([x, y, z])
  
    np.save("pred_poses.npy", np.array(pred_poses))

if __name__ == "__main__":
    main()