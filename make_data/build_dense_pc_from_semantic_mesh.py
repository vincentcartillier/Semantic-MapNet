import os
import json
import h5py
import torch
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
from utils.habitat_utils import HabitatUtils
from utils.semantic_utils import label_colours
from config import use_fine, object_whitelist
from config import resolution

env_splits = json.load(open('data/envs_splits.json', 'r'))
envs = env_splits['envs']

output_dir = 'data/point_clouds/'

for env in tqdm(envs):

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

    semantic_colors = {oid:label_colours[sid] for oid, sid in semantic_ids.items()}

    vertices = []
    obj_ids  = []
    sem_ids  = []
    colors   = []

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
            obj_ids.append(oid)
            sem_ids.append(semantic_ids[oid])
            colors.append(semantic_colors[oid])

            vertices.append(p2)
            obj_ids.append(oid)
            sem_ids.append(semantic_ids[oid])
            colors.append(semantic_colors[oid])

            vertices.append(p3)
            obj_ids.append(oid)
            sem_ids.append(semantic_ids[oid])
            colors.append(semantic_colors[oid])


            n1 = (p2 - p1)
            d1 = np.linalg.norm(n1)
            if d1 == 0: continue
            n1 = n1 / d1

            n2 = (p3 - p1)
            d2 = np.linalg.norm(n2)
            if d2 == 0: continue
            n2 = n2 / d2

            for i in np.arange(0, d1, 0.01):

                b = (d1-i) * d2/d1

                for j in np.arange(0, b, 0.01):

                    p = p1 + i*n1 + j*n2

                    vertices.append(p)
                    obj_ids.append(oid)
                    sem_ids.append(semantic_ids[oid])
                    colors.append(semantic_colors[oid])

    vertices = np.asarray(vertices)
    obj_ids  = np.array(obj_ids)
    sem_ids  = np.array(sem_ids)
    sem_ids  = sem_ids.astype(np.int32)
    colors   = np.array(colors)

    filename = os.path.join(output_dir, env+'.h5')
    with h5py.File(filename, 'w') as f:
        f.create_dataset('vertices', data=vertices, dtype=np.float)
        f.create_dataset('obj_ids', data=obj_ids, dtype=np.int32)
        f.create_dataset('sem_ids', data=sem_ids, dtype=np.int32)
        f.create_dataset('colors', data=colors, dtype=np.uint8)

    del habitat

