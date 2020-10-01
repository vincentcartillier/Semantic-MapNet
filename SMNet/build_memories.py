import os
import yaml
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import convert_weights_cuda_cpu

from model_test_memories import SMNet

split = 'test'

data_dir = 'data/{}_data/'.format(split)
expe_dir = 'runs/gru_fullrez_lastlayer_m256_DDP/88116'

output_dir = os.path.join(expe_dir, 'memories', split)
Path(output_dir).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- load configs
with open(os.path.join(expe_dir, 'smnet.yml')) as fp:
    cfg = yaml.load(fp)


# -- create model
model_path = os.path.join(expe_dir, 'smnet_mp3d_best_model.pkl')
model = SMNet(cfg['model'], device)
model = model.to(device)

print('Loading pre-trained weights: ', model_path)
state = torch.load(model_path)
model_state = state['model_state']
model_state = convert_weights_cuda_cpu(model_state, 'cpu')
model.load_state_dict(model_state)
model.eval()


# -- load JSONS and select envs
info = json.load(open('data/GT/semmap_GT_info.json','r'))
paths = json.load(open('data/paths.json', 'r'))
envs_splits = json.load(open('data/envs_splits.json', 'r'))
envs = envs_splits['{}_envs'.format(split)]
envs = [x for x in envs if x in paths]
envs.sort()



with torch.no_grad():
    for env in envs:

        if os.path.isfile(os.path.join(output_dir, env+'.h5')): continue

        # get env dim
        world_dim_discret = info[env]['dim']
        map_height = world_dim_discret[2]
        map_width  = world_dim_discret[0]

        # load DATA
        h5file = h5py.File(os.path.join(data_dir, 'projections', env+'.h5'), 'r')
        projections_wtm = np.array(h5file['proj_world_to_map'], dtype=np.uint16)
        mask_outliers = np.array(h5file['mask_outliers'], dtype=np.bool)
        heights = np.array(h5file['heights'], dtype=np.float32)
        h5file.close()

        h5file = h5py.File(os.path.join(data_dir, 'features', env+'.h5'), 'r')
        if cfg['data']['feature_type'] == 'encoder':
            features = np.array(h5file['features_encoder'], dtype=np.float32)
        elif cfg['data']['feature_type'] == 'lastlayer':
            features = np.array(h5file['features_lastlayer'], dtype=np.float32)
        elif cfg['data']['feature_type'] == 'scores':
            features = np.array(h5file['features_scores'], dtype=np.float32)
        elif cfg['data']['feature_type'] == 'softmax':
            features = np.array(h5file['features_scores'], dtype=np.float32)
        elif cfg['data']['feature_type'] == 'onehot':
            features = np.array(h5file['features_scores'], dtype=np.float32)
        else:
            raise Exception('{} feature type not supported.'.format(cfg['data']['feature_type']))
        h5file.close()

        features = torch.from_numpy(features)

        if cfg['data']['feature_type'] == 'softmax':
            features = torch.nn.functional.softmax(features, dim=1)

        if cfg['data']['feature_type'] == 'onehot':
            features = features.permute(0,2,3,1)
            num_classes = features.size(3)
            labels = features.max(3)[1]
            features = torch.nn.functional.one_hot(labels, num_classes=num_classes)
            features = features.float()
            features = features.permute(0,3,1,2)


        projections_wtm = projections_wtm.astype(np.int32)
        projections_wtm = torch.from_numpy(projections_wtm)
        mask_outliers = torch.from_numpy(mask_outliers)
        heights = torch.from_numpy(heights)

        memory = model(features,
                       projections_wtm,
                       mask_outliers,
                       heights,
                       map_height,
                       map_width)


        memory = memory.cpu().numpy()
        memory = memory.astype(np.float32)

        filename = os.path.join(output_dir, file)
        with h5py.File(filename, 'w') as f:
            f.create_dataset('memory', data=memory, dtype=np.float32)



