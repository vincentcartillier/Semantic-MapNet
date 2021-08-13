import os

import h5py
import numpy as np
from imageio import imwrite
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from tqdm import tqdm

folder_pred = "../ObjectNav/semmap/"
output_dir = "../ObjectNav/freespace_map/"

envs = os.listdir(folder_pred)

envs = [x.split(".")[0] for x in envs]

for env in tqdm(envs):

    file = h5py.File(os.path.join(folder_pred, env + ".h5"), "r")
    semmap_pred = np.array(file["semmap"])
    height_map = np.array(file["height_map"])
    height_map[height_map > 0] = (
        height_map[height_map > 0] - height_map[height_map > 0].min() + 1
    )
    file.close

    n, h = np.histogram(height_map[height_map > 0], bins=200)
    ih = np.argmax(n)
    floor_height = h[ih]

    nav_map = (height_map > floor_height - 0.05) & (height_map < floor_height + 0.05)

    nav_map = binary_closing(nav_map.astype(int), structure=np.ones((5, 5))).astype(
        np.bool
    )

    nav_map = binary_erosion(nav_map.astype(int), structure=np.ones((10, 10))).astype(
        np.bool
    )
    nav_map = binary_dilation(nav_map.astype(int), structure=np.ones((5, 5))).astype(
        np.bool
    )

    nav_map = nav_map.astype(np.uint8)
    nav_map *= 255
    nav_map = nav_map.astype(np.uint8)

    filename = os.path.join(output_dir, env + ".png")
    imwrite(filename, nav_map)
