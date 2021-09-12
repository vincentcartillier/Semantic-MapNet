import os
import sys
import json
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append('../metric/')
from metric.iou import IoU

sys.path.append('../utils/')
from semantic_utils import object_whitelist

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


if dataset == 'mp3d':
    metrics = IoU(13)
elif dataset == 'replica':
    metrics = IoU(13, ignore_index=5)
metrics.reset()

total = 0

filename = os.path.join(pred_dir, 'evaluation_metrics.h5')
with h5py.File(filename, 'w') as f:
    for env in tqdm(envs):

        file = env+'.h5'

        if not os.path.isfile(os.path.join(pred_dir, 'semmap', file)): continue

        total += 1

        gt_h5_file = h5py.File(os.path.join(GT_dir, file), 'r')
        gt_semmap = np.array(gt_h5_file['map_semantic'])
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

        obj_gt = gt_semmap[observed_map]
        obj_pred = pred_semmap[observed_map]

        f.create_dataset('{}_pred'.format(env), data=obj_pred, dtype=np.int16)
        f.create_dataset('{}_gt'.format(env), data=obj_gt, dtype=np.int16)

        metrics.add(obj_pred, obj_gt)


print('total #envs= ', total, '\n')

classes_iou, mIoU, acc, recalls, mRecall, precisions, mPrecision = metrics.value()

print('Mean IoU: ', "%.2f" % round(mIoU*100, 2))
print('Overall Acc: ', "%.2f" % round(acc*100, 2))
print('Mean Recall: ',  "%.2f" % round(mRecall*100, 2))
print('Mean Precision: ', "%.2f" % round(mPrecision*100, 2))

print('\n per class IoU:')
for i in range(13):
    print('      ',  "%.2f" % round(classes_iou[i]*100, 2), object_whitelist[i])
print('\n Precision:')
for i in range(13):
    print('      ',  "%.2f" % round(precisions[i]*100, 2), object_whitelist[i])
print('\n Recall:')
for i in range(13):
    print('      ',  "%.2f" % round(recalls[i]*100, 2), object_whitelist[i])


