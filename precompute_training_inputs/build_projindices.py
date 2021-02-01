import os
import json
import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch_scatter import scatter_max

from multiprocessing import Pool

output_name = 'smnet_training_data_maxHIndices' 
output_root = 'data/training'
output_dir = os.path.join(output_root, 
                          output_name)
files = os.listdir('data/training/smnet_training_data/')

device = torch.device('cpu')

def get_projections_indices(file):
    h5file = h5py.File(os.path.join('data/training/smnet_training_data', file), 'r')
    point_clouds = np.array(h5file['projection_indices'])
    heights = point_clouds[:,:,:,1]
    h5file.close()
    
    h5file = h5py.File(os.path.join('data/training/smnet_training_data_indices', file), 'r')
    proj_indices = np.array(h5file['indices'])
    masks_outliers = np.array(h5file['masks_outliers'])
    h5file.close()

    heights = torch.from_numpy(heights).float()
    proj_indices = torch.from_numpy(proj_indices).long()
    masks_outliers = torch.from_numpy(masks_outliers).bool()

    heights = heights.to(device)
    proj_indices = proj_indices.to(device)
    masks_outliers = masks_outliers.to(device)

    masks_inliers = ~masks_outliers
    
    argmax_indices = []
    for t in range(20):
        mask_inliers = masks_inliers[t,:,:]
        proj_index = proj_indices[t,:,:,:]
        height = heights[t,:,:]
        
        flat_indices = proj_index[mask_inliers, :]
        height = height[mask_inliers]
        height += 1000
        assert (height > 0).all()
        
        
        feat_index = 250 * flat_indices[:, 1] + flat_indices[:, 0]
        feat_index = feat_index.long()
        
        # -- projection
        flat_highest_z = torch.zeros(250*250, device=device)
        flat_highest_z, argmax_flat_spatial_map = scatter_max(
            height,
            feat_index,
            dim=0,
            out = flat_highest_z,
        )

        argmax_indices.append(argmax_flat_spatial_map.cpu().numpy())
    
    argmax_indices = np.asarray(argmax_indices)
    masks_outliers = masks_outliers.cpu().numpy()
    filename = os.path.join(output_dir, file)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('indices', data=np.array(argmax_indices), dtype=np.int32)
        f.create_dataset('masks_outliers', data=masks_outliers, dtype=np.bool)
    

print(' Start collapsing projections indices (Projecting the pixel with highest height in case of collision.)')

pool = Pool(40)
res = pool.map(get_projections_indices, files) 

print(' -> Done')

print('Build .h5 files for each splits')

envs_splits = json.load(open('../data/envs_splits.json', 'r'))

files = os.listdir(output_dir)

for split in ['train', 'val']:
    projection_indices = []
    projection_masks = []
    projection_indices_envs = []
    for file in files:
        name = file.split('.')[0]
        env = '_'.join(name.split('_')[:2])
        if env in envs_splits['{}_envs'.format(split)]:
            h5file = h5py.File(os.path.join(output_dir, file))
            indices = np.array(h5file['indices'])
            indices = indices[np.newaxis, ...]
            projection_indices.append(indices)
            mask_outliers = np.array(h5file['masks_outliers'])
            mask_outliers = mask_outliers[np.newaxis, ...]
            projection_masks.append(mask_outliers)
            projection_indices_envs.append(name)
            h5file.close()
    
    projection_indices = np.concatenate(projection_indices, axis=0)
    projection_masks = np.concatenate(projection_masks, axis=0)
    with h5py.File(os.path.join(output_root, 
                                output_name+'_{}.h5'.format(split)), 
                   'w') as f:
        f.create_dataset('indices', data=np.array(projection_indices), dtype=np.int32)
        f.create_dataset('masks_outliers', data=projection_masks, dtype=np.bool)
 
    json.dump(projection_indices_envs, 
              open(os.path.join(output_root,
                                output_name+'_{}.json'.format(split)), 
                   'w'))

