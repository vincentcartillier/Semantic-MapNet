"""
Builds a colored point cloud from a semantic mesh (.ply).
Colors are assigned based on the semantic object id
"""
import os, sys
import numpy as np
import open3d as o3d
from tqdm import tqdm
from plyfile import PlyData

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../'))

from utils.habitat_utils import HabitatUtils
from utils.semantic_utils import label_colours
from utils.semantic_utils import use_fine, object_whitelist

env = '17DRP5sb8fy_0'

sampling_resolution = 0.01

house, level = env.split('_')
scene = 'data/mp3d/{}/{}.glb'.format(house, house)
habitat = HabitatUtils(scene, int(level))

objects = habitat.get_objects_in_level()
objects = habitat.keep_objects_in_whitelist(objects)

objects_ids = list(objects.keys())

semantic_ids = {}
for oid in objects_ids:
    object_name = objects[oid].category.name(mapping='mpcat40')
    if object_name in use_fine:
        object_name = objects[oid].category.name(mapping='raw')
    semantic_id = object_whitelist.index(object_name)+1
    semantic_ids[oid] = semantic_id

colors = {oid:label_colours[sid] for oid, sid in semantic_ids.items()}

vertices = []
vertices_color = []

ply_file = 'data/mp3d/{}/{}_semantic.ply'.format(house, house)
ply_data = PlyData.read(ply_file)

for face in tqdm(ply_data['face']):
    vids = list(face[0])
    oid = face[1]
    if oid in objects_ids:
        p1 = ply_data['vertex'][vids[0]]
        p1 = np.array([p1[0], p1[2], -p1[1]])
        p2 = ply_data['vertex'][vids[1]]
        p2 = np.array([p2[0], p2[2], -p2[1]])
        p3 = ply_data['vertex'][vids[2]]
        p3 = np.array([p3[0], p3[2], -p3[1]])

        vertices.append(p1)
        vertices_color.append(colors[oid])
        vertices.append(p2)
        vertices_color.append(colors[oid])
        vertices.append(p3)
        vertices_color.append(colors[oid])


        n1 = (p2 - p1)
        d1 = np.linalg.norm(n1)
        n1 = n1 / d1

        n2 = (p3 - p1)
        d2 = np.linalg.norm(n2)
        n2 = n2 / d2

        for i in np.arange(0, d1, sampling_resolution):

            b = (d1-i) * d2/d1

            for j in np.arange(0, b, sampling_resolution):

                p = p1 + i*n1 + j*n2

                vertices.append(p)
                vertices_color.append(colors[oid])

vertices = np.array(vertices)
vertices_color = np.array(vertices_color)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)
point_cloud.colors = o3d.utility.Vector3dVector(vertices_color)

filename = 'data/point_clouds/{}.ply'.format(env)
o3d.io.write_point_cloud(filename, point_cloud)

