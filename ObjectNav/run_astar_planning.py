import bisect
import json
import os
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
from astar import Astar, Node
from imageio import imread
from scipy.ndimage import binary_closing, binary_dilation, binary_opening
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils.semantic_utils import object_whitelist

resolution = 0.02


data_dir = "../data/ObjectNav/objectnav_mp3d_v1/val/"
output_dir = "../data/ObjectNav/"

all_goals = json.load(open("ObjNav_GT_goals.json", "r"))


def run_astar(episode):

    MAX_RUNS = 2000000
    TURN_ANGLE = 30
    FORWARD_STEP_SIZE = 0.25
    AGENT_HEIGHT = 0.88
    AGENT_RADIUS = 0.18
    house_floor_threshold = {
        "2azQ1b91cZZ": [1.5],
        "8194nk5LbLH": [1.2],
        "EU6Fwq7SyZv": [-1.3, 2.0],
        "oLBMNvg9in8": [-0.8, 1.8, 4.6],
        "pLe4wQe7qrG": [],
        "QUCTc6BB5sX": [-1.5],
        "TbHJrupSAjP": [-1.5, 1.7],
        "X7HyMhZNoso": [1.8],
        "x8F5xyUWy9e": [],
        "Z6MFQCViBuw": [],
        "zsNo4HB9uLZ": [],
    }

    # -- folders
    data_dir = "../data/ObjectNav/objectnav_mp3d_v1/val/"
    folder_pred = "../data/ObjectNav/semmap/"
    floormap_dir = "../data/ObjectNav/freespace_map/"

    info = json.load(open("../data/ObjectNav/semmap_objnav_info.json", "r"))

    # -- setup naming bindings
    read_tsv = csv.reader(open("mpcat40.tsv"), delimiter="\t")
    mpcat40 = {line[0]: line[1] for line in read_tsv}

    jsonfile = json.load(open(os.path.join(data_dir, "val.json"), "r"))
    category_to_mp3d_category_id = jsonfile["category_to_mp3d_category_id"]

    house = episode["scene_id"].split("/")[1]

    # -- sample object category target
    obj_semantic_id = category_to_mp3d_category_id[episode["object_category"]]
    obj_semantic_name = mpcat40[str(obj_semantic_id)]
    if obj_semantic_name in object_whitelist:
        sid = object_whitelist.index(obj_semantic_name) + 1
    else:
        return (episode["episode_id"], [], [])

    # -- select floor -> env
    if house_floor_threshold[house]:
        level = bisect.bisect(
            house_floor_threshold[house], episode["start_position"][1]
        )
    else:
        level = 0

    env = house + "_" + str(level)

    map_world_shift = np.array(info[env]["map_world_shift"])

    # -- load maps
    floormap = imread(os.path.join(floormap_dir, env + ".png"))
    floormap = floormap.astype(np.float)
    floormap /= 255
    floormap = floormap.astype(np.bool)

    if not os.path.isfile(os.path.join(folder_pred, env + ".h5")):
        return (episode["episode_id"], [], [])

    h5file = h5py.File(os.path.join(folder_pred, env + ".h5"), "r")
    map_semantic = np.array(h5file["semmap"])
    observed_map = np.array(h5file["observed_map"])
    observed_map = observed_map.astype(np.float)
    h5file.close()

    map_semantic = np.multiply(map_semantic, observed_map)
    map_semantic = map_semantic.astype(np.int)

    # goals = None
    goals = all_goals[house][episode["episode_id"]]
    goals = np.array(goals)
    goals -= map_world_shift
    goals = goals[:, [0, 2]]

    # -- get init position
    start_pos = episode["start_position"]
    start_pos -= map_world_shift

    start_x = start_pos[0]
    start_y = start_pos[2]
    start_row = int(np.round(start_y / resolution))
    start_col = int(np.round(start_x / resolution))

    start_rot = np.array(episode["start_rotation"])
    r = R.from_quat(start_rot)
    _, start_heading, _ = r.as_rotvec()

    start_heading = start_heading * 180 / np.pi
    start_heading = -(start_heading + 90)

    start = Node(
        x=start_x, y=start_y, heading=start_heading, row=start_row, col=start_col
    )

    # -- get maps for Astar
    goal_mask = map_semantic == sid
    goal_mask = binary_opening(goal_mask.astype(int), structure=np.ones((3, 3))).astype(
        np.bool
    )
    floormap = binary_closing(floormap.astype(int), structure=np.ones((10, 10))).astype(
        np.bool
    )
    navmap = floormap & (map_semantic == 0)

    # compute Heuristic
    # -- Euclidean distance
    non_object_mask = ~goal_mask
    non_object_mask = non_object_mask.astype(np.uint8)
    distance_map = cv2.distanceTransform(non_object_mask.copy(), cv2.DIST_L2, 3)

    # -- Geodesic distance
    # os.system('pip install scikit-fmm')
    # import skfmm
    # import numpy.ma as ma
    # mask = ~navmap&~goal_mask
    # mask = ~mask
    # mask = binary_dilation(mask.astype(int), structure=np.ones((10,10))).astype(np.bool)
    # mask = ~mask
    # map = np.ones(navmap.shape)
    # map = map - 2*goal_mask.astype(np.float)
    # map = ma.masked_array(map, mask)
    # distance_map = skfmm.distance(map)

    pathfinder = Astar(
        navmap,
        observed_map,
        heuristic=distance_map,
        init_heading=start_heading,
        goals=goals,
    )

    path, runs = pathfinder.run(start, goal_mask, max_runs=MAX_RUNS)

    if len(path) == 0:
        actions = []
    else:
        path = path[::-1]

        # -- convert path to actions
        actions = []
        prev_p = path[0]
        for p in path[1:]:
            # -- rotate
            pre_h = prev_p[2]
            new_h = p[2]

            delta_h = (new_h - pre_h + 360) % 360

            if delta_h == 0:
                pass
            elif delta_h <= 180:
                # trun right
                num_rotations = int(delta_h / TURN_ANGLE)
                actions += [3] * abs(num_rotations)
            elif delta_h > 180:
                # trun left
                delta_h = 360 - delta_h
                num_rotations = int(delta_h / TURN_ANGLE)
                actions += [2] * abs(num_rotations)

            # move forward
            actions.append(1)

            prev_p = p

        actions.append(0)

    return (episode["episode_id"], actions, path)


pool = Pool(32)


files = [
    "2azQ1b91cZZ.json",
    "8194nk5LbLH.json",
    "EU6Fwq7SyZv.json",
    "oLBMNvg9in8.json",
    "pLe4wQe7qrG.json",
    "QUCTc6BB5sX.json",
    "TbHJrupSAjP.json",
    "X7HyMhZNoso.json",
    "x8F5xyUWy9e.json",
    "Z6MFQCViBuw.json",
    "zsNo4HB9uLZ.json",
]

outputs = {}

for file in tqdm(files):

    house = file.split(".")[0]

    outputs[house] = {}

    jsonfile = json.load(open(os.path.join(data_dir, "content", file), "r"))

    res = pool.map(run_astar, jsonfile["episodes"])

    for i, r in enumerate(res):

        outputs[house][i] = {"actions": r[1], "path": r[2]}


json.dump(outputs, open(os.path.join(output_dir, "astar_planning_outputs.json"), "w"))
