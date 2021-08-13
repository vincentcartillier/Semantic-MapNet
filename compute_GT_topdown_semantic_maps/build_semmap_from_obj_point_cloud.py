import json
import os

import h5py
import numpy as np
import torch
from torch_scatter import scatter_max
from tqdm import tqdm

resolution = 0.02

obj_point_clouds = "data/object_point_clouds/"

obj_files = os.listdir(obj_point_clouds)

houses_dim = json.load(open("data/houses_dim.json", "r"))

output_dir = "data/"
info = {}

for obj_f in tqdm(obj_files):

    env = obj_f.split(".")[0]

    f = h5py.File(os.path.join(obj_point_clouds, obj_f), "r")
    vertices = np.array(f["vertices"])
    obj_ids = np.array(f["obj_ids"])
    sem_ids = np.array(f["sem_ids"])
    colors = np.array(f["colors"])
    f.close()

    # --- change coordinates to match map
    # --  set discret dimensions
    center = np.array(houses_dim[env]["center"])
    sizes = np.array(houses_dim[env]["sizes"])
    sizes += 2  # -- pad env bboxes

    world_dim = sizes.copy()
    world_dim[1] = 0

    central_pos = center.copy()
    central_pos[1] = 0

    map_world_shift = central_pos - world_dim / 2

    world_dim_discret = [
        int(np.round(world_dim[0] / resolution)),
        0,
        int(np.round(world_dim[2] / resolution)),
    ]

    info[env] = {
        "dim": world_dim_discret,
        "central_pos": [float(x) for x in central_pos],
        "map_world_shift": [float(x) for x in map_world_shift],
    }

    # -- some maps have 0 obj of interest
    if len(vertices) == 0:
        info[env]["y_min_value"] = 0.0
        mask = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.bool)
        map_z = np.zeros((world_dim_discret[2], world_dim_discret[0]), dtype=np.float32)
        map_instance = np.zeros(
            (world_dim_discret[2], world_dim_discret[0]), dtype=np.int32
        )
        map_semantic = np.zeros(
            (world_dim_discret[2], world_dim_discret[0]), dtype=np.int32
        )
        filename = os.path.join(output_dir, env + ".h5")
        with h5py.File(filename, "w") as f:
            f.create_dataset("mask", data=mask, dtype=np.bool)
            f.create_dataset("map_z", data=map_z, dtype=np.float32)
            f.create_dataset("map_instance", data=map_instance, dtype=np.int32)
            f.create_dataset("map_semantic", data=map_semantic, dtype=np.int32)
        continue

    vertices -= map_world_shift

    # -- discretize point cloud
    vertices = torch.FloatTensor(vertices)
    obj_ids = torch.FloatTensor(obj_ids)
    sem_ids = torch.FloatTensor(sem_ids)

    y_values = vertices[:, 1]

    vertex_to_map_x = (vertices[:, 0] / resolution).round()
    vertex_to_map_z = (vertices[:, 2] / resolution).round()

    outside_map_indices = (
        (vertex_to_map_x >= world_dim_discret[0])
        + (vertex_to_map_z >= world_dim_discret[2])
        + (vertex_to_map_x < 0)
        + (vertex_to_map_z < 0)
    )

    # assert outside_map_indices.sum() == 0
    y_values = y_values[~outside_map_indices]
    vertex_to_map_z = vertex_to_map_z[~outside_map_indices]
    vertex_to_map_x = vertex_to_map_x[~outside_map_indices]

    obj_ids = obj_ids[~outside_map_indices]
    sem_ids = sem_ids[~outside_map_indices]

    # -- get the z values for projection
    # -- shift to positive values
    min_y = y_values.min()
    y_values = y_values - min_y
    y_values += 1.0

    info[env]["y_min_value"] = float(min_y.item())

    # -- projection
    feat_index = (world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x).long()
    flat_highest_z = torch.zeros(int(world_dim_discret[0] * world_dim_discret[2]))
    flat_highest_z, argmax_flat_spatial_map = scatter_max(
        y_values,
        feat_index,
        dim=0,
        out=flat_highest_z,
    )

    m = argmax_flat_spatial_map >= 0
    flat_map_instance = (
        torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
    )
    flat_map_instance[m.view(-1)] = obj_ids[argmax_flat_spatial_map[m]]

    flat_map_semantic = torch.zeros(int(world_dim_discret[0] * world_dim_discret[2]))
    flat_map_semantic[m.view(-1)] = sem_ids[argmax_flat_spatial_map[m]]

    # -- format data
    mask = m.reshape(world_dim_discret[2], world_dim_discret[0])
    mask = mask.numpy()
    mask = mask.astype(np.bool)
    map_z = flat_highest_z.reshape(world_dim_discret[2], world_dim_discret[0])
    map_z = map_z.numpy()
    map_z = map_z.astype(np.float)
    map_instance = flat_map_instance.reshape(world_dim_discret[2], world_dim_discret[0])
    map_instance = map_instance.numpy()
    map_instance = map_instance.astype(np.float)
    map_semantic = flat_map_semantic.reshape(world_dim_discret[2], world_dim_discret[0])
    map_semantic = map_semantic.numpy()
    map_semantic = map_semantic.astype(np.float)

    filename = os.path.join(output_dir, "semmap", env + ".h5")
    with h5py.File(filename, "w") as f:
        f.create_dataset("mask", data=mask, dtype=np.bool)
        f.create_dataset("map_heights", data=map_z, dtype=np.float32)
        f.create_dataset("map_instance", data=map_instance, dtype=np.int32)
        f.create_dataset("map_semantic", data=map_semantic, dtype=np.int32)

json.dump(info, open(os.path.join(output_dir, "semmap_GT_info.json"), "w"))
