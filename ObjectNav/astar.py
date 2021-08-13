import heapq

import cv2
import geopandas as gpd
import numpy as np
from geocube.api.core import make_geocube
from matplotlib import pyplot as plt
from shapely.geometry import LineString

TURN_ANGLE = 30
FORWARD_STEP_SIZE = 0.25
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.02
AGENT_RADIUS_MARGIN = 0.01

ANGLES = [x * TURN_ANGLE for x in range(12)]


class Node(object):
    def __init__(
        self,
        x=0.0,
        y=0.0,
        heading=0.0,
        heading_id=0,
        row=0,
        col=0,
    ):

        # real world coordinates
        self.x = x
        self.y = y
        self.heading = heading
        self.heading_id = heading_id

        # closest pixels on the topdown map
        self.row = row
        self.col = col

        # scores
        self.h = np.inf
        self.g = np.inf
        self.f = np.inf

        # in openSet flag
        self.opened = False

    def __lt__(self, other):
        return self.f < other.f


class Astar:
    def __init__(
        self,
        navmap,
        observed_map,
        heuristic,
        init_heading=0.0,
        resolution=0.02,
        goals=None,
    ):

        self.navmap = navmap
        self.observed_map = observed_map
        self.heuristic = heuristic
        self.goalmap = np.zeros(navmap.shape, dtype=np.bool)

        self.resolution = resolution

        self.init_heading = init_heading

        # -- init circle mask of agent's radius
        self.agent_radius = AGENT_RADIUS + AGENT_RADIUS_MARGIN  # 10cm + 2cm margin

        # -- init agent path mask on 10cm for every angles
        self.angles = {i: angle + self.init_heading for i, angle in enumerate(ANGLES)}

        self.goals = goals

        self.path_masks = {}
        for i, angle in self.angles.items():

            line = LineString(
                [
                    (0, 0),
                    (
                        np.cos((angle) * np.pi / 180.0) * FORWARD_STEP_SIZE,
                        -np.sin((angle) * np.pi / 180.0) * FORWARD_STEP_SIZE,
                    ),
                ]
            )

            dilated = line.buffer(self.agent_radius, cap_style=1)

            gdf = gpd.GeoDataFrame({"mask": [1]}, geometry=[dilated], crs="EPSG:4326")
            cube = make_geocube(
                gdf, resolution=(-self.resolution, self.resolution), fill=0
            )
            arr_mask = cube.mask.values == 1

            self.path_masks[i] = arr_mask

        # -- init circle mask of agent's 1m radius vision
        self.agent_vision_radius = 1.0 - 0.2  # 1m 80cm
        MH = int(2 * self.agent_vision_radius / self.resolution)
        if MH % 2 == 0:
            MH += 1
        self.MH = MH
        self.vision_circle_mask = np.zeros((MH, MH), dtype=np.bool)
        C = int(np.floor(MH / 2))
        self.C = C
        for i in range(MH):
            for j in range(MH):
                d = ((i - C) * self.resolution) ** 2 + ((j - C) * self.resolution) ** 2
                if d <= self.agent_vision_radius ** 2:
                    self.vision_circle_mask[i, j] = True

    def path_success(self, current, neighbor):
        curr_r = current.row
        curr_c = current.col
        neig_r = neighbor.row
        neig_c = neighbor.col
        heading_id = neighbor.heading_id

        center_r = int(np.round((curr_r + neig_r) / 2))
        center_c = int(np.round((curr_c + neig_c) / 2))

        mask = self.path_masks[heading_id]

        size_r, size_c = mask.shape

        size_r_h = int(size_r / 2)
        size_c_h = int(size_c / 2)

        roi = self.navmap[
            center_r - size_r_h : center_r - size_r_h + size_r,
            center_c - size_c_h : center_c - size_c_h + size_c,
        ]

        if not roi.shape == mask.shape:
            return False

        roi_masked = roi[mask]
        if not roi_masked.all():
            return False

        return True

    def plot_neighbors(self, current, neighbor, map, ax):
        curr_x = current.x
        curr_y = current.y
        heading = neighbor.heading

        map = cv2.circle(map.copy(), (current.col, current.row), 1, (255, 0, 0))
        for d in [2.0, 4.0, 6.0, 8.0, 10.0]:
            x = curr_x + np.cos(heading * np.pi / 180.0) * d / 100.0
            y = curr_y + np.sin(heading * np.pi / 180.0) * d / 100.0
            row = int(np.round(y / self.resolution))
            col = int(np.round(x / self.resolution))

            map = cv2.circle(map.copy(), (col, row), self.c, (0, 0, 255))

        ax.imshow(map)
        plt.pause(0.3)
        plt.draw()
        return map

    def plot_current_node(self, current, map, ax):

        map = cv2.circle(map.copy(), (current.col, current.row), 2, (255, 0, 0), -1)

        ax.imshow(map)
        plt.pause(0.03)
        plt.draw()
        map = cv2.circle(map.copy(), (current.col, current.row), 2, (0, 0, 255), -1)
        return map

    def is_goal(self, current):
        if self.goals is not None:

            curr_pos = np.array([current.x, current.y])

            test = self.goals - curr_pos
            test = test ** 2
            test = np.sum(test, axis=1)
            test = np.sqrt(test)

            if (test <= 0.02).any():
                return True

            return False

        else:
            row = current.row
            col = current.col
            roi_goal = self.goalmap[
                row - self.C : row + self.C + 1, col - self.C : col + self.C + 1
            ]
            roi_obse = self.observed_map[
                row - self.C : row + self.C + 1, col - self.C : col + self.C + 1
            ]

            # roi dim (2xC+1, 2xC+1)
            if roi_goal.shape != self.vision_circle_mask.shape:
                return False

            roi_goal = np.multiply(roi_goal, self.vision_circle_mask)
            indices_row, indices_col = np.nonzero(roi_goal)
            if len(indices_row) == 0:
                return False
            else:
                for ir, ic in zip(indices_row, indices_col):
                    tmp_r = self.C + (ir - self.C) * np.linspace(0, 1, 100)
                    tmp_c = self.C + (ic - self.C) * np.linspace(0, 1, 100)

                    tmp_r = np.round(tmp_r)
                    tmp_c = np.round(tmp_c)

                    tmp_r = tmp_r.astype(np.int)
                    tmp_c = tmp_c.astype(np.int)

                    ray_obse = roi_obse[tmp_r, tmp_c]

                    if ray_obse.all():
                        return True
            return False

    def reconstruct_path(self, cameFrom, current):
        path = [[current.x, current.y, current.heading]]
        node = current
        while node in cameFrom:
            node = cameFrom[node]
            path.append([node.x, node.y, node.heading])
        return path

    def run(self, start, goalmap, max_runs=10000, map=None, ax=None):

        self.goalmap = goalmap

        start.h = self.heuristic[start.row, start.col]
        start.g = 0.0
        start.f = self.heuristic[start.row, start.col]
        start.opened = True

        runs = 0
        openSet = []
        heapq.heappush(openSet, start)

        all_nodes = {(start.row, start.col): start}

        cameFrom = {}

        while len(openSet) > 0:

            runs += 1

            if runs > max_runs:
                return [], runs

            current = heapq.heappop(openSet)
            current.opened = False

            if ax is not None:
                map = self.plot_current_node(current, map, ax)

            if self.is_goal(current):
                return self.reconstruct_path(cameFrom, current), runs

            curr_x = current.x
            curr_y = current.y

            # loop through neighboors
            for angle_id, angle in self.angles.items():

                neighbor_heading = angle
                neighbor_x = (
                    curr_x
                    + np.cos(neighbor_heading * np.pi / 180.0) * FORWARD_STEP_SIZE
                )
                neighbor_y = (
                    curr_y
                    + np.sin(neighbor_heading * np.pi / 180.0) * FORWARD_STEP_SIZE
                )

                neighbor_row = int(np.round(neighbor_y / self.resolution))
                neighbor_col = int(np.round(neighbor_x / self.resolution))

                if (neighbor_row, neighbor_col) in all_nodes:
                    neighbor = all_nodes[(neighbor_row, neighbor_col)]
                else:
                    neighbor = Node(
                        x=neighbor_x,
                        y=neighbor_y,
                        heading=neighbor_heading,
                        heading_id=angle_id,
                        row=neighbor_row,
                        col=neighbor_col,
                    )
                    all_nodes[(neighbor_row, neighbor_col)] = neighbor

                if self.path_success(current, neighbor):

                    tentative_gScore = current.g + FORWARD_STEP_SIZE

                    # if ax is not None:
                    #    map = self.plot_neighbors(current, neighbor, map, ax)

                else:
                    # node unreachable
                    tentative_gScore = current.g + np.inf
                    continue

                if tentative_gScore < neighbor.g:
                    cameFrom[neighbor] = current
                    neighbor.g = tentative_gScore
                    neighbor.f = neighbor.g + self.heuristic[neighbor.row, neighbor.col]
                    neighbor.heading = neighbor_heading
                    neighbor.heading_id = angle_id
                    if not neighbor.opened:
                        neighbor.opened = True
                        heapq.heappush(openSet, neighbor)

        return [], runs
