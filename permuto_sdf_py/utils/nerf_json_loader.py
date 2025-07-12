import json
import numpy as np
import os

class NeRFJsonLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.frames = []
        self.intrinsics = None
        self.load_json()

    def load_json(self):
        json_path = os.path.join(self.dataset_dir, "transforms_train.json")
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.frames = meta["frames"]
        # Example: get intrinsics from meta if available, or set defaults
        self.intrinsics = meta.get("fl_x", None), meta.get("fl_y", None), meta.get("cx", None), meta.get("cy", None)

    def get_image_paths(self):
        return [os.path.join(self.dataset_dir, frame["file_path"] + ".png") for frame in self.frames]

    def get_poses(self):
        return [np.array(frame["transform_matrix"]) for frame in self.frames]

    def get_intrinsics(self):
        return self.intrinsics