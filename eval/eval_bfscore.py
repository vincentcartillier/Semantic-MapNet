import os
import sys
import json
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append('../utils/')
from semantic_utils import object_whitelist


from bfscore import bfscore

split = 'test'
dataset = 'replica'

object_whitelist = ['void'] + object_whitelist

if dataset == 'mp3d':
    GT_dir = 'data/GT/semmap/'
    obsmaps_dir = 'data/observed_masks'
elif dataset == 'replica':
    GT_dir = 'data/replica/semmap/'
    obsmaps_dir = 'data/replica/observed_masks'


# -- select prediction dir
pred_dir = 'data/replica/OUTPUTS/fullrez/SMNet_gru_lastlayer_m256/'

if dataset == 'mp3d':
    paths = json.load(open('data/paths.json', 'r'))
    envs_splits = json.load(open('data/envs_splits.json', 'r'))
    envs = envs_splits['{}_envs'.format(split)]
    envs = [x for x in envs if x in paths]
    envs.sort()
elif dataset == 'replica':
    paths = json.load(open('../replica/paths.json', 'r'))
    envs = list(paths.keys())
    envs.sort()
    envs.remove('room_2')



def compute_bfscore(env):

    file = env+'.h5'

    if not os.path.isfile(os.path.join(pred_dir, 'semmap', file)):
        return env, False,0,0,0,0

    gt_h5_file = h5py.File(os.path.join(GT_dir, file), 'r')
    gt_semmap = np.array(gt_h5_file['map_semantic']).astype(np.float)
    gt_h5_file.close()

    pred_h5_file = h5py.File(os.path.join(pred_dir, 'semmap', file), 'r')
    if 'map_semantic' in pred_h5_file:
        pred_semmap = np.array(pred_h5_file['map_semantic'])
    else:
        pred_semmap = np.array(pred_h5_file['semmap'])
    pred_h5_file.close()

    h5file = h5py.File(os.path.join(obsmaps_dir, file), 'r')
    observed_map = np.array(h5file['observed_map'])
    observed_map = observed_map.astype(np.bool)
    h5file.close()

    obj_gt = np.multiply(gt_semmap, observed_map)
    obj_gt = obj_gt.astype(np.uint8)
    obj_pred = np.multiply(pred_semmap, observed_map)
    obj_pred = obj_pred.astype(np.uint8)
    _, _, pn, pd, rn, rd = bfscore(obj_pred, obj_gt, 13, threshold=3)

    return env, True, pn, pd, rn, rd



pool = Pool(21)
res = pool.map(compute_bfscore, envs)


total = 0

precision_num = np.zeros(12)
precision_den = np.zeros(12)
recall_num = np.zeros(12)
recall_den = np.zeros(12)

envs_bf1 = {}


for r in res:

    env, isfile, pn, pd, rn, rd = r

    if isfile:
        total += 1
        precision_num += pn
        precision_den += pd
        recall_num += rn
        recall_den += rd

        envs_bf1[env] = {}
        envs_bf1[env]['pn'] = [int(x) for x in pn]
        envs_bf1[env]['pd'] = [int(x) for x in pd]
        envs_bf1[env]['rn'] = [int(x) for x in rn]
        envs_bf1[env]['rd'] = [int(x) for x in rd]



json.dump(envs_bf1, open(os.path.join(pred_dir, 'evaluation_boundaryF1.json'), 'w'))

precision = precision_num / precision_den
recall = recall_num / recall_den

bf1 = 2*recall*precision/(recall + precision)    # boundary F1 score

if dataset == 'replica':
    bf1[4] = np.nan

mbf1 = np.nanmean(bf1)

print('total #envs= ', total, '\n')

print('Mean Boundary F1: ', "%.2f" % round(mbf1*100, 2))

print('\n F1:')
for i in range(12):
    print('      ',  "%.2f" % round(bf1[i]*100, 2), object_whitelist[i+1])



