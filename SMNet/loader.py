import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils import data

envs_splits = json.load(open("data/envs_splits.json", "r"))


class SMNetLoader(data.Dataset):
    def __init__(self, cfg, split="train"):

        self.split = split
        self.root = cfg["root"]
        self.ego_downsample = cfg["ego_downsample"]
        self.feature_type = cfg["feature_type"]

        self.files = os.listdir(os.path.join(self.root, "smnet_training_data"))

        self.files = [
            x
            for x in self.files
            if "_".join(x.split("_")[:2]) in envs_splits["{}_envs".format(split)]
        ]
        self.envs = [x.split(".")[0] for x in self.files]

        # -- load semantic map GT
        h5file = h5py.File(
            os.path.join(self.root, "smnet_training_data_semmap.h5"), "r"
        )
        self.semmap_GT = np.array(h5file["semantic_maps"])
        h5file.close()
        self.semmap_GT_envs = json.load(
            open(os.path.join(self.root, "smnet_training_data_semmap.json"), "r")
        )
        self.semmap_GT_indx = {
            i: self.semmap_GT_envs.index(self.envs[i] + ".h5")
            for i in range(len(self.files))
        }

        # -- load projection indices
        if self.ego_downsample:
            h5file = h5py.File(
                os.path.join(
                    self.root,
                    "smnet_training_data_maxHIndices_every4_{}.h5".format(split),
                ),
                "r",
            )
            self.projection_indices = np.array(h5file["indices"])
            self.masks_outliers = np.array(h5file["masks_outliers"])
            h5file.close()
            self.projection_indices_envs = json.load(
                open(
                    os.path.join(
                        self.root,
                        "smnet_training_data_maxHIndices_every4_{}.json".format(split),
                    ),
                    "r",
                )
            )
        else:
            h5file = h5py.File(
                os.path.join(
                    self.root, "smnet_training_data_maxHIndices_{}.h5".format(split)
                ),
                "r",
            )
            self.projection_indices = np.array(h5file["indices"])
            self.masks_outliers = np.array(h5file["masks_outliers"])
            h5file.close()
            self.projection_indices_envs = json.load(
                open(
                    os.path.join(
                        self.root,
                        "smnet_training_data_maxHIndices_{}.json".format(split),
                    ),
                    "r",
                )
            )

        self.projection_indices_indx = {
            i: self.projection_indices_envs.index(self.envs[i])
            for i in range(len(self.files))
        }

        assert len(self.files) > 0

        self.available_idx = list(range(len(self.files)))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]

        h5file = h5py.File(os.path.join(self.root, "smnet_training_data", file), "r")
        if self.feature_type == "encoder":
            features = np.array(h5file["features_encoder"])
        elif self.feature_type == "lastlayer":
            features = np.array(h5file["features_lastlayer"])
        elif self.feature_type == "scores":
            features = np.array(h5file["features_scores"])
        elif self.feature_type == "softmax":
            features = np.array(h5file["features_scores"])
        elif self.feature_type == "onehot":
            features = np.array(h5file["features_scores"])
        else:
            raise Exception("{} feature type not supported.".format(self.feature_type))
        h5file.close()

        features = torch.from_numpy(features).float()
        if self.feature_type == "softmax":
            features = torch.nn.functional.softmax(features, dim=1)

        if self.feature_type == "onehot":
            features = features.permute(0, 2, 3, 1)
            num_classes = features.size(3)
            labels = features.max(3)[1]
            features = F.one_hot(labels, num_classes=num_classes)
            features = features.bool()
            features = features.permute(0, 3, 1, 2)
        else:
            features = features.half()

        projection_index = self.projection_indices_indx[env_index]
        proj_indices = self.projection_indices[projection_index]
        masks_outliers = self.masks_outliers[projection_index]

        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()

        masks_inliers = ~masks_outliers

        semmap_index = self.semmap_GT_indx[env_index]
        semmap = self.semmap_GT[semmap_index]
        semmap = torch.from_numpy(semmap).long()

        return (features, masks_inliers, proj_indices, semmap, file)
