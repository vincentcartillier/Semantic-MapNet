import torch
import torch.nn as nn
import numpy as np

from projector.core import ProjectorUtils


class PointCloud(ProjectorUtils):
    """
    Unprojects 2D depth pixels in 3D
    """

    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        world_shift_origin,
        z_clip_threshold,
        device=torch.device("cuda"),
    ):
        """Init function

        Args:
            vfov (float): Vertical Field of View
            batch_size (float)
            feature_map_height (int): height of image
            feature_map_width (int): width of image
            world_shift_origin (float, float, float): (x, y, z) shift apply to position the map in the world coordinate system.
            z_clip_threshold (float): in meters. Pixels above camera height + z_clip_threshold will be ignored. (mainly ceiling pixels)
            device (torch.device, optional): Defaults to torch.device('cuda').
        """

        ProjectorUtils.__init__(self, 
                                vfov, 
                                batch_size, 
                                feature_map_height,
                                feature_map_width, 
                                1, 
                                1,
                                1, 
                                world_shift_origin, 
                                z_clip_threshold,
                                device)
        
        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.world_shift_origin = world_shift_origin
        self.z_clip_threshold = z_clip_threshold
        self.device = device


    def forward(self, depth, T, obs_per_map=1):
        """Forward Function

        Args:
            depth (torch.FloatTensor): Depth image
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)
            obs_per_map (int): obs_per_map images are projected to the same map

        Returns:
            mask (torch.FloatTensor): mask of outliers. Mainly when no depth is present.
            point cloud (torch.FloatTensor) 

        """

        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        # -- filter out the semantic classes with depth == 0. Those sem_classes map to the agent
        # itself .. and thus are considered outliers
        no_depth_mask = depth == 0

        # Feature mappings in the world coordinate system where origin is somewhere but not camera
        # # GEO:
        # shape: features_to_world (N, features_height, features_width, 3)
        point_cloud = self.pixel_to_world_mapping(depth, T)

        return point_cloud, no_depth_mask

