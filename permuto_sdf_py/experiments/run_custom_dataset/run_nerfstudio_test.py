#!/usr/bin/env python3

# This script is used to train PermutoSDF on a dataset with OPENCV camera model format (direct intrinsics).

###RUN WITH
'''
python3 /workspace/permuto_sdf/permuto_sdf_py/experiments/run_custom_dataset/run_nerfstudio_test.py \
  --dataset custom \
  --dataset_path /workspace/datasets/new_tomato_soup_can/data \
  --exp_info opencv_test1 \
  --no_viewer
'''

import torch
import argparse
import os
import natsort
import numpy as np
import json

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import TrainParams
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.permuto_sdf_utils import get_frames_cropped
from permuto_sdf_py.train_permuto_sdf import train
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes


torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


parser = argparse.ArgumentParser(description='Train sdf and color')
parser.add_argument('--dataset', default="custom", help='Dataset name which can also be custom in which case the user has to provide their own data')
parser.add_argument('--dataset_path', default="/workspace/datasets/new_tomato_soup_can/data", help='Dataset path')
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
parser.add_argument('--scene_scale', default=0.9, type=float, help='Scale of the scene so that it fits inside the unit sphere')
parser.add_argument('--scene_translation', default=[0,0,0], type=float, nargs=3, help='Translation of the scene so that it fits inside the unit sphere')
parser.add_argument('--img_subsample', default=1.0, type=float, help="The higher the subsample value, the smaller the images are. Useful for low vram")
args = parser.parse_args()
with_viewer=not args.no_viewer


SCENE_SCALE=args.scene_scale
SCENE_TRANSLATION=args.scene_translation
IMG_SUBSAMPLE_FACTOR=args.img_subsample #subsample the image to lower resolution in case you are running on a low VRAM GPU. The higher this number, the smaller the images
DATASET_PATH=args.dataset_path 
 
def create_custom_dataset():
    """
    Loads images and camera parameters from a JSON file with OPENCV camera model,
    applies coordinate transforms, and prepares frames for training.
    This handles the OPENCV convention where intrinsics are directly specified
    as fl_x, fl_y, cx, cy, w, h at the top level.
    """
    json_path = os.path.join(DATASET_PATH, "transforms.json") # load the json file with the camera poses and intrinsics
    with open(json_path, 'r') as f:
        meta = json.load(f)

    frames = []
    
    # Extract camera intrinsics from top level (OPENCV format)
    if 'fl_x' in meta and 'fl_y' in meta:
        # Direct focal length specification
        fl_x = meta['fl_x']
        fl_y = meta['fl_y'] 
        cx = meta.get('cx', None)
        cy = meta.get('cy', None)
        w = meta.get('w', None)
        h = meta.get('h', None)
        print(f"Using OPENCV format: fx={fl_x}, fy={fl_y}, cx={cx}, cy={cy}, w={w}, h={h}")
    elif 'camera_angle_x' in meta:
        # Fallback to FOV format
        camera_angle_x = meta['camera_angle_x']
        use_fov = True
        print(f"Using camera_angle_x: {camera_angle_x}")
    else:
        raise ValueError("No camera intrinsics found in JSON file")
    
    for idx, frame_data in enumerate(meta['frames']):
        # Handle different file path formats
        file_path = frame_data['file_path']
        
        # Try different path combinations
        possible_paths = [
            os.path.join(DATASET_PATH, file_path),
            os.path.join(DATASET_PATH, file_path.lstrip('./')),  # Remove leading ./
        ]
        
        # If no extension, try adding .png
        if not file_path.endswith(('.png', '.jpg', '.jpeg')):
            possible_paths.extend([
                os.path.join(DATASET_PATH, file_path + ".png"),
                os.path.join(DATASET_PATH, file_path + ".jpg"),
                os.path.join(DATASET_PATH, file_path + ".jpeg"),
            ])
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
                
        if img_path is None:
            print(f"Warning: Could not find image for {file_path}")
            continue
            
        print("img_name", img_path)
        frame = Frame()

        # Load image
        img = Mat(img_path)
        img = img.to_cv32f()
        if img.channels() == 4:
            img_rgb = img.rgba2rgb()
        else:
            img_rgb = img
        frame.rgb_32f = img_rgb
        frame.width = img.cols
        frame.height = img.rows

        # Optionally load mask
        if args.with_mask and img.channels() == 4:
            img_mask = img.get_channel(3)
            frame.mask = img_mask
        elif args.with_mask:
            print("Warning: Mask requested but image does not have alpha channel")

        # Set up camera intrinsics
        K = np.identity(3)
        
        if 'fl_x' in meta and 'fl_y' in meta:
            # Use direct focal length specification (OPENCV format)
            K[0][0] = fl_x  # fx
            K[1][1] = fl_y  # fy
            K[0][2] = cx if cx is not None else img.cols / 2  # cx
            K[1][2] = cy if cy is not None else img.rows / 2  # cy
        else:
            # Calculate focal length from FOV (fallback)
            focal = 0.5 * img.cols / np.tan(0.5 * camera_angle_x)
            K[0][0] = focal  # fx
            K[1][1] = focal  # fy
            K[0][2] = img.cols / 2  # cx
            K[1][2] = img.rows / 2  # cy
        
        frame.K = K

        # Extrinsics - transform_matrix is cam-to-world
        tf_world_cam_np = np.array(frame_data['transform_matrix'])  # cam-to-world

        # Rotation matrices for coordinate system conversion
        theta_x = np.radians(90)
        rot_x_90 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x), 0],
            [0, np.sin(theta_x), np.cos(theta_x), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Apply coordinate transformation (NeRF/Blender to OpenCV)
        coord_transform = np.diag([1, -1, -1, 1])
        tf_world_cam_np = tf_world_cam_np @ coord_transform
        
        # Convert cam-to-world to world-to-cam
        tf_cam_world_np = np.linalg.inv(tf_world_cam_np)
        tf_cam_world_np = tf_cam_world_np @ rot_x_90 

        frame.tf_cam_world.from_matrix(tf_cam_world_np.astype(np.float32))

        # Scale/Translate scene to fit in unit sphere
        tf_world_cam_rescaled = frame.tf_cam_world.inverse()
        translation = tf_world_cam_rescaled.translation().copy()
        translation *= SCENE_SCALE
        translation += SCENE_TRANSLATION
        tf_world_cam_rescaled.set_translation(translation)
        frame.tf_cam_world = tf_world_cam_rescaled.inverse()

        # Subsample image if requested
        frame = frame.subsample(IMG_SUBSAMPLE_FACTOR)

        # Create frustum for visualization
        frustum_mesh = frame.create_frustum_mesh(scale_multiplier=0.06)
        Scene.show(frustum_mesh, "frustum_mesh_" + str(idx))

        frames.append(frame)

    return frames


def run():

    config_file="train_permuto_sdf.cfg"
    config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)
    train_params=TrainParams.create(config_path)
    hyperparams=HyperParamsPermutoSDF()


    #get the checkpoints path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints/custom_dataset")
    os.makedirs(checkpoint_path, exist_ok=True)

    
    train_params.set_with_tensorboard(True)
    train_params.set_save_checkpoint(True)
    print("checkpoint_path",checkpoint_path)
    print("with_viewer", with_viewer)

    experiment_name="opencv_custom"
    if args.exp_info:
        experiment_name+="_"+args.exp_info
    print("experiment name",experiment_name)


    #CREATE CUSTOM DATASET---------------------------
    frames=create_custom_dataset() 

    #print the scale of the scene which contains all the cameras.
    print("scene centroid", Scene.get_centroid()) #aproximate center of our scene which consists of all frustum of the cameras
    print("scene scale", Scene.get_scale()) #how big the scene is as a measure betwen the min and max of call cameras positions

    ##VISUALIZE
    # view=Viewer.create()
    # while True:
        # view.update()


    ####train
    tensor_reel=MiscDataFuncs.frames2tensors(frames) #make an tensorreel and get rays from all the images at 
    train(args, config_path, hyperparams, train_params, None, experiment_name, with_viewer, checkpoint_path, tensor_reel, frames_train=frames, hardcoded_cam_init=False)


def main():
    run()


if __name__ == "__main__":
     main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')
