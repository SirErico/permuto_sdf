#!/usr/bin/env python3

#this scripts shows how to run PermutoSDF on your own custom dataset
#You would need to modify the function create_custom_dataset() to suit your needs. The current code is setup to read from the easypbr_render dataset (see README.md for the data) but you need to change it for your own data. The main points are that you need to provide an image, intrinsics and extrinsics for each your cameras. Afterwards you need to scale your scene so that your object of interest lies within the bounding sphere of radius 0.5 at the origin.

#CALL with ./permuto_sdf_py/experiments/run_custom_dataset/run_custom_dataset.py --exp_info test [--no_viewer]

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
from permuto_sdf_py.utils.nerf_json_loader import NeRFJsonLoader



torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


parser = argparse.ArgumentParser(description='Train sdf and color')
parser.add_argument('--dataset', default="custom", help='Dataset name which can also be custom in which case the user has to provide their own data')
parser.add_argument('--dataset_path', default="/media/rosu/Data/data/permuto_sdf_data/easy_pbr_renders/head/", help='Dataset path')
parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
parser.add_argument('--no_viewer', action='store_true', help="Set this to true in order disable the viewer")
parser.add_argument('--scene_scale', default=0.9, type=float, help='Scale of the scene so that it fits inside the unit sphere')
parser.add_argument('--scene_translation', default=[0,0,0], type=float, nargs=3, help='Translation of the scene so that it fits inside the unit sphere')
parser.add_argument('--img_subsample', default=1.0, type=float, help="The higher the subsample value, the smaller the images are. Useful for low vram")
args = parser.parse_args()
with_viewer=not args.no_viewer


#MODIFY these for your dataset!
SCENE_SCALE=args.scene_scale
SCENE_TRANSLATION=args.scene_translation
IMG_SUBSAMPLE_FACTOR=args.img_subsample #subsample the image to lower resolution in case you are running on a low VRAM GPU. The higher this number, the smaller the images
DATASET_PATH=args.dataset_path #point this to wherever you downloaded the easypbr_data (see README.md for download link)
 
def create_custom_dataset():
    json_path = os.path.join(DATASET_PATH, "transforms_train.json")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    frames = []
    for idx, frame_data in enumerate(meta['frames']):
        img_path = os.path.join(DATASET_PATH, frame_data['file_path'] + ".png")
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

        if args.with_mask and img.channels() == 4:
            img_mask = img.get_channel(3)
            frame.mask = img_mask
        elif args.with_mask:
            exit("Mask requested but image does not have alpha channel")

        # Intrinsics from FOV
        camera_angle_x = meta['camera_angle_x']
        focal = 0.5 * img.cols / np.tan(0.5 * camera_angle_x)
        K = np.identity(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = img.cols / 2
        K[1][2] = img.rows / 2
        frame.K = K

        # Extrinsics
        tf_world_cam_np = np.array(frame_data['transform_matrix'])  # cam-to-world

        # Rotation matrices
        theta_x = np.radians(90)
        rot_x_90 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x), 0],
            [0, np.sin(theta_x), np.cos(theta_x), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        coord_transform = np.diag([1, -1, -1, 1])
        # Apply coordinate transformation
        tf_world_cam_np = tf_world_cam_np @ coord_transform
        tf_cam_world_np = np.linalg.inv(tf_world_cam_np)
        tf_cam_world_np = tf_cam_world_np @ rot_x_90 

        frame.tf_cam_world.from_matrix(tf_cam_world_np.astype(np.float32))

        # Scale/Translate scene
        tf_world_cam_rescaled = frame.tf_cam_world.inverse()
        translation = tf_world_cam_rescaled.translation().copy()
        translation *= SCENE_SCALE
        translation += SCENE_TRANSLATION
        tf_world_cam_rescaled.set_translation(translation)
        frame.tf_cam_world = tf_world_cam_rescaled.inverse()

        # Subsample
        frame = frame.subsample(IMG_SUBSAMPLE_FACTOR)

        # Frustum for visualization
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

    experiment_name="custom"
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
