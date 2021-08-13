"""
The purpose of this script is to give intuitions on
how to build the GT topdown semantic maps by projecting
egocentric GT semantic labels.

Note: In the paper we didn't use this strategy, rather
we projected the semantic mesh directly.
Check : build_semmap_from_obj_point_cloud.py
"""

import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "../"))

from scipy.spatial.transform import Rotation as R
from torch_scatter import scatter_add, scatter_max

from projector import PointCloud, _transform3D
from utils.habitat_utils import HabitatUtils
from utils.semantic_utils import color_label

"""

    settings

"""

env = "17DRP5sb8fy_0"

resolution = 0.02

vfov = 67.5
vfov = vfov * np.pi / 180.0

z_clip = 0.50

features_spatial_dimensions = (480, 640)


"""

    some util fonctions

"""


def is_in_bbox(points, o):
    N = len(points)
    T = o.obb.world_to_local
    half_extents = o.obb.half_extents
    p = np.ones((N, 4))
    p[:, :3] = points
    p_local = np.matmul(T, p.T)
    p_local = p_local.T
    p_local = p_local[:, :3]
    p_local = p_local * half_extents

    return np.all(
        ((-half_extents - 0.1) <= p_local) & ((half_extents + 0.1) >= p_local), axis=1
    )


device = torch.device("cuda")

paths = json.load(open("data/paths.json", "r"))

info = json.load(open("data/semmap_GT_info.json", "r"))

house, level = env.split("_")
scene = "data/mp3d/{}/{}.glb".format(house, house)
habitat = HabitatUtils(scene, int(level))


objects = habitat.get_objects_in_house()
objects = habitat.keep_objects_in_whitelist(objects)


world_shift_origin = torch.FloatTensor([0, 0, 0]).to(device=device)

projector = PointCloud(
    vfov,
    1,
    features_spatial_dimensions[0],
    features_spatial_dimensions[1],
    world_shift_origin,
    z_clip,
    device=device,
)

path = paths[env]

N = len(path["positions"])

point_cloud = []
semantics = []
instances = []

world_dim_discret = info[env]["dim"]
map_world_shift = info[env]["map_world_shift"]
map_world_shift = torch.FloatTensor(map_world_shift).to(device=device)
observed_map = torch.zeros(
    (world_dim_discret[2], world_dim_discret[0]), dtype=torch.bool
)

with torch.no_grad():
    for n in tqdm(range(N)):
        pos = path["positions"][n]
        ori = path["orientations"][n]

        habitat.position = list(pos)
        habitat.rotation = list(ori)
        habitat.set_agent_state()

        sensor_pos = habitat.get_sensor_pos()
        sensor_ori = habitat.get_sensor_ori()

        # -- get T transorm
        sensor_ori = np.array([sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w])
        r = R.from_quat(sensor_ori)
        elevation, heading, bank = r.as_rotvec()

        xyzhe = np.array(
            [[sensor_pos[0], sensor_pos[1], sensor_pos[2], heading, elevation + np.pi]]
        )
        xyzhe = torch.FloatTensor(xyzhe).to(device)
        T = _transform3D(xyzhe, device=device)

        # -- depth for projection
        depth = habitat.render(mode="depth")
        depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        depth *= 10.0
        depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

        # -- projection
        pc, mask = projector.forward(depth_var, T)

        # -- get semantic labels
        semantic = habitat.render_semantic_12cat()
        instance = habitat.render(mode="semantic")
        sem_mask = semantic > 0

        # -- get mask of observed locations
        tmp_pc = pc.clone()
        tmp_pc = tmp_pc.view(-1, 3)
        tmp_pc -= map_world_shift
        vertex_to_map_x = (tmp_pc[:, 0] / resolution).round()
        vertex_to_map_z = (tmp_pc[:, 2] / resolution).round()

        outside_map_indices = (
            (vertex_to_map_x >= world_dim_discret[0])
            + (vertex_to_map_z >= world_dim_discret[2])
            + (vertex_to_map_x < 0)
            + (vertex_to_map_z < 0)
        )

        feat_index = (world_dim_discret[0] * vertex_to_map_z + vertex_to_map_x).long()
        feat_index = feat_index[~outside_map_indices]
        ones = torch.ones(feat_index.shape).to(device=device)
        flat_curr_observed_map = torch.zeros(
            int(world_dim_discret[0] * world_dim_discret[2])
        ).to(device=device)
        flat_curr_observed_map = scatter_add(
            ones,
            feat_index,
            dim=0,
            out=flat_curr_observed_map,
        )

        curr_observed_map = flat_curr_observed_map.reshape(
            world_dim_discret[2], world_dim_discret[0]
        ).to("cpu")
        curr_observed_map = curr_observed_map > 0
        observed_map += curr_observed_map

        # store relevant 3D points
        pc = pc[0].cpu().numpy()
        mask = mask[0].cpu().numpy()

        # --  maskout inliers
        mask = ~mask & sem_mask

        pc = pc[mask]
        semantic = semantic[mask]
        instance = instance[mask]

        # filter points using GT bboxes
        unique_instances = np.unique(instance)
        for oid in unique_instances:
            if not oid in objects:
                continue
            mask_instance = instance == oid
            pc_instance = pc[mask_instance]
            sem_instance = semantic[mask_instance]
            ins_instance = instance[mask_instance]

            in_bbox_indices = is_in_bbox(pc_instance, objects[oid])

            if in_bbox_indices.any():
                pc_instance = pc_instance[in_bbox_indices]
                sem_instance = sem_instance[in_bbox_indices]
                ins_instance = ins_instance[in_bbox_indices]

                point_cloud.append(pc_instance)
                semantics.append(sem_instance)
                instances.append(ins_instance)


observed_map = observed_map.numpy()
observed_map = observed_map.astype(np.bool)

point_cloud = np.concatenate(point_cloud, axis=0)
semantics = np.concatenate(semantics, axis=0)
instances = np.concatenate(instances, axis=0)


# -- discretize point cloud
point_cloud = torch.FloatTensor(point_cloud)
point_cloud -= map_world_shift.cpu()
instances = torch.FloatTensor(instances.astype(np.float32))
semantics = torch.FloatTensor(semantics.astype(np.float32))

y_values = point_cloud[:, 1]

vertex_to_map_x = (point_cloud[:, 0] / resolution).round()
vertex_to_map_z = (point_cloud[:, 2] / resolution).round()

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

instances = instances[~outside_map_indices]
semantics = semantics[~outside_map_indices]


# -- get the y values for projection
# -- shift to positive values
min_y = y_values.min()
y_values = y_values - min_y
y_values += 1.0

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
flat_map_instance = torch.zeros(int(world_dim_discret[0] * world_dim_discret[2])) - 1
flat_map_instance[m.view(-1)] = instances[argmax_flat_spatial_map[m]]

flat_map_semantic = torch.zeros(int(world_dim_discret[0] * world_dim_discret[2]))
flat_map_semantic[m.view(-1)] = semantics[argmax_flat_spatial_map[m]]

# -- format data
map_instance = flat_map_instance.reshape(world_dim_discret[2], world_dim_discret[0])
map_instance = map_instance.numpy()
map_instance = map_instance.astype(np.int)
map_semantic = flat_map_semantic.reshape(world_dim_discret[2], world_dim_discret[0])
map_semantic = map_semantic.numpy()
map_semantic = map_semantic.astype(np.int)

map_semantic_color = color_label(map_semantic)
map_semantic_color = map_semantic_color.transpose(1, 2, 0)
map_semantic_color = map_semantic_color.astype(np.uint8)


import matplotlib.pyplot as plt

plt.subplot(131)
plt.imshow(map_semantic_color)
plt.title("Topdown semantic map")
plt.axis("off")
plt.subplot(132)
plt.imshow(map_instance)
plt.title("Topdown instance map")
plt.axis("off")
plt.subplot(133)
plt.imshow(observed_map)
plt.title("Topdown observed map")
plt.axis("off")
plt.show()
