import os
import sys
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))
from projector import _transform3D
from projector.projector import Projector
from utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R


output_dir = 'data/test_data/projections/'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda')

#Settings
resolution = 0.02 # topdown resolution
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

# -- -- Load json
paths = json.load(open('data/paths.json', 'r'))

info = json.load(open('data/semmap_GT_info.json', 'r'))

envs_splits = json.load(open('data/envs_splits.json', 'r'))
test_envs = envs_splits['test_envs']
test_envs = [x for x in test_envs if x in paths]
test_envs.sort()

for env in test_envs:

    # -- instantiate Habitat
    house, level = env.split('_')
    scene = 'data/mp3d/{}/{}.glb'.format(house, house)
    habitat = HabitatUtils(scene, int(level))

    # -- get house info
    world_dim_discret = info[env]['dim']
    map_world_shift = info[env]['map_world_shift']
    map_world_shift = np.array(map_world_shift)
    world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)

    # -- instantiate projector
    projector = Projector(vfov, 1,
                          default_ego_dim[0],
                          default_ego_dim[1],
                          world_dim_discret[2], # height
                          world_dim_discret[0], # width
                          resolution,
                          world_shift_origin,
                          z_clip,
                          device=device)

    path = paths[env]

    N = len(path['positions'])

    projections_wtm = np.zeros((N,480,640,2), dtype=np.uint16)
    projections_masks = np.zeros((N,480,640), dtype=np.bool)
    projections_heights = np.zeros((N,480,640), dtype=np.float32)

    with torch.no_grad():
        for n in tqdm(range(N)):
            pos = path['positions'][n]
            ori = path['orientations'][n]

            habitat.position = list(pos)
            habitat.rotation = list(ori)
            habitat.set_agent_state()

            sensor_pos = habitat.get_sensor_pos()
            sensor_ori = habitat.get_sensor_ori()

            # -- get T transorm
            sensor_ori = np.array([sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w])
            r = R.from_quat(sensor_ori)
            elevation, heading, bank = r.as_rotvec()

            xyzhe = np.array([[sensor_pos[0],
                               sensor_pos[1],
                               sensor_pos[2],
                               heading,
                               elevation + np.pi]])

            xyzhe = torch.FloatTensor(xyzhe).to(device)
            T = _transform3D(xyzhe, device=device)

            # -- depth for projection
            depth = habitat.render(mode='depth')
            depth = depth[:,:,0]
            depth = depth.astype(np.float32)
            depth *= 10.0
            depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

            # -- projection
            world_to_map, mask_outliers, heights = projector.forward(depth_var, T, return_heights=True)

            world_to_map = world_to_map[0].cpu().numpy()
            mask_outliers = mask_outliers[0].cpu().numpy()
            heights = heights[0].cpu().numpy()

            world_to_map = world_to_map.astype(np.uint16)
            mask_outliers = mask_outliers.astype(np.bool)
            heights = heights.astype(np.float32)

            projections_wtm[n,...] = world_to_map
            projections_masks[n,...] = mask_outliers
            projections_heights[n,...] = heights

    filename = os.path.join(output_dir, env+'.h5')
    with h5py.File(filename, 'w') as f:
        f.create_dataset('proj_world_to_map', data=projections_wtm, dtype=np.uint16)
        f.create_dataset('mask_outliers', data=projections_masks, dtype=np.bool)
        f.create_dataset('heights', data=projections_heights, dtype=np.float32)

    del habitat, projector
