#!/usr/bin/env python3

#this scripts runs in a serial way through all the checkpoints that we have in list_of_checkpoints and creates meshes out of them

#DTU meshes are in another frame that does not correspond with the ground truth but we want to transform it so that it corresponds
#more info on the evaluation is here
# https://github.com/Totoro97/NeuS/issues/43
#for running Neus you need to run:
# python exp_runner.py --mode validate_mesh --conf ./confs/wmask.conf --case "dtu_scan24" --is_continue
# python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case "bvms_bear" --is_continue
#and you need to modify conf/wmask to point to the DTU dataset and the checkpoints path

###RUN WITH
'''
python3 create_my_meshes_nerf.py \
  --dataset custom \
  --comp_name comp_3 \
  --res 1000 \
  --scan_name mic \
  --ckpt_path /workspace/permuto_sdf/checkpoints/custom_dataset/custom_mic/200000/models \
  --output_name my_mic_mesh
'''

import torch

import sys
import os
import numpy as np
from tqdm import tqdm
import time
import torchvision
import argparse

import easypbr
from easypbr  import *
from dataloaders import *

import permuto_sdf
from permuto_sdf  import TrainParams
from permuto_sdf  import OccupancyGrid
from permuto_sdf_py.models.models import SDF
from permuto_sdf_py.models.models import RGB
from permuto_sdf_py.models.models import NerfHash
from permuto_sdf_py.models.models import Colorcal
from permuto_sdf_py.utils.sdf_utils import extract_mesh_from_sdf_model
from permuto_sdf_py.utils.common_utils import create_dataloader
from permuto_sdf_py.utils.common_utils import create_bb_for_dataset
from permuto_sdf_py.utils.common_utils import nchw2lin
from permuto_sdf_py.utils.common_utils import lin2nchw
from permuto_sdf_py.utils.permuto_sdf_utils import load_from_checkpoint
from permuto_sdf_py.train_permuto_sdf import run_net_in_chunks
from permuto_sdf_py.train_permuto_sdf import HyperParamsPermutoSDF
import permuto_sdf_py.paths.list_of_checkpoints as list_chkpts
import permuto_sdf_py.paths.list_of_training_scenes as list_scenes

from permuto_sdf_py.utils.nerf_json_loader import NeRFJsonLoader
import open3d as o3d


config_file="train_permuto_sdf.cfg"

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_grad_enabled(False)
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../../../config', config_file)



def extract_mesh_and_transform_to_original_tf(model, nr_points_per_dim, loader, aabb):

    if isinstance(model, SDF):
        extracted_mesh=extract_mesh_from_sdf_model(model, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5)
    elif isinstance(model, INGP):
        extracted_mesh=extract_mesh_from_density_model(model, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5, threshold=40)
    elif isinstance(model, SDFNetwork):
        extracted_mesh=extract_mesh_from_sdf_model_neus(model, nr_points_per_dim=nr_points_per_dim, min_val=-0.5, max_val=0.5)
        

    # extracted_mesh=aabb.remove_points_outside(extracted_mesh)
    #remove points outside the aabb
    points=torch.from_numpy(extracted_mesh.V).float().cuda()
    is_valid=aabb.check_point_inside_primitive(points)
    extracted_mesh.remove_marked_vertices( is_valid.flatten().bool().cpu().numpy() ,True)
    extracted_mesh.recalculate_min_max_height()
   
    #transform the extracted mesh from the easypbr coordinate frame to the dtu one so that it matches the gt
    if isinstance(loader, DataLoaderDTU):
        tf_easypbr_dtu=loader.get_tf_easypbr_dtu()
        tf_dtu_easypbr=tf_easypbr_dtu.inverse()
        extracted_mesh.transform_model_matrix(tf_dtu_easypbr.to_double())
        extracted_mesh.apply_model_matrix_to_cpu(True)

    return extracted_mesh



def run():
    #argparse
    parser = argparse.ArgumentParser(description='prepare dtu evaluation')
    parser.add_argument('--dataset', required=True, default="", help="dataset which can be dtu or bmvs or custom")
    parser.add_argument('--comp_name', required=True, help='Tells which computer are we using which influences the paths for finding the data')
    parser.add_argument('--res', required=True, help="Resolution of the mesh, usually at least 700")
    parser.add_argument('--with_mask', action='store_true', help="Set this to true in order to train with a mask")
    parser.add_argument('--scan_name', required=True, help="Name of the scan or scene (e.g. custom_lego, mask_mic)")
    parser.add_argument('--ckpt_path', required=True, help="Full path to the checkpoint directory (ending with /models)")
    parser.add_argument('--output_name', required=True, help="Name for the output mesh file (without extension)")
    args = parser.parse_args()
    hyperparams=HyperParamsPermutoSDF()


    #get the results path which will be at the root of the permuto_sdf package 
    permuto_sdf_root=os.path.dirname(os.path.abspath(permuto_sdf.__file__))
    results_path=os.path.join(permuto_sdf_root, "results")
    os.makedirs(results_path, exist_ok=True)
    # ckpts
    checkpoint_path=os.path.join(permuto_sdf_root, "checkpoints")



    #####PARAMETERS#######
    with_viewer=False
    print("====PARAMETERS====")
    print("args.dataset", args.dataset)
    print("args.with_mask", args.with_mask)
    print("results_path",results_path)
    print("with_viewer", with_viewer)
    print("args.res", args.res)
    print("args.scan_name", args.scan_name)
    print("args.ckpt_path", args.ckpt_path)
    print("args.output_name", args.output_name)
    print("====================")
    iter_nr_for_anneal=9999999
    cos_anneal_ratio=1.0
    low_res=False
    nr_points_per_dim=int(args.res) #can go up to 2300


    aabb = create_bb_for_dataset(args.dataset)
    

    #params for rendering
    model_sdf=SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.sdf_nr_iters_for_c2f).to("cuda")
    model_rgb=RGB(in_channels=3, boundary_primitive=aabb, geom_feat_size_in=hyperparams.sdf_geom_feat_size, nr_iters_for_c2f=hyperparams.rgb_nr_iters_for_c2f).to("cuda")
    model_bg=NerfHash(4, boundary_primitive=aabb, nr_iters_for_c2f=hyperparams.background_nr_iters_for_c2f ).to("cuda") 
    if hyperparams.use_occupancy_grid:
        occupancy_grid=OccupancyGrid(256, 1.0, [0,0,0])
    else:
        occupancy_grid=None
    model_sdf.train(False)
    model_rgb.train(False)
    model_bg.train(False)

    scan_name = args.scan_name
    ckpt_path_full = args.ckpt_path
    output_name = args.output_name

    # Use NeRFJsonLoader for custom dataset, otherwise use create_dataloader
    if args.dataset == "custom":
        nerf_data_path = f"/workspace/nerf_synthetic/{scan_name}"
        loader = NeRFJsonLoader(nerf_data_path)
    else:
        loader, _ = create_dataloader(config_path, args.dataset, scan_name, low_res, args.comp_name, args.with_mask)

    load_from_checkpoint(ckpt_path_full, model_sdf, model_rgb, model_bg, occupancy_grid)

    extracted_mesh = extract_mesh_and_transform_to_original_tf(model_sdf, nr_points_per_dim, loader, aabb)

    out_mesh_path = os.path.join(permuto_sdf_root, "results/output_permuto_sdf_meshes", args.dataset, "custom")
    os.makedirs(out_mesh_path, exist_ok=True)
    
    mesh_output_path = os.path.join(out_mesh_path, output_name + ".ply")
    extracted_mesh.save_to_file(mesh_output_path)

    print(f"Mesh saved to {mesh_output_path}")

    # Check mesh
    try: 
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_output_path)
        print("Mesh loaded successfully with Open3D.")
        print("Number of vertices:", len(o3d_mesh.vertices))
        print("Number of triangles:", len(o3d_mesh.triangles))
    except Exception as e:
        print(f"Failed to load mesh with Open3D: {e}")

    return


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