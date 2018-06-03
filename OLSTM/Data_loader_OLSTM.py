import os
import pandas
import torch
from OLSTM_utils import *


class DataLoader:

    def __init__(self, batch_size, samples_to_load, data_path, scaling_factor=10, grid_size=16, max_dist_neighbors=60, verbose=False):
        self.batch_size = batch_size
        self.index = 0
        self.scaling_factor = scaling_factor
        self.grid_size = grid_size
        self.max_dist_neighbors = max_dist_neighbors
        self.verbose = verbose
        self.load_data(samples_to_load, data_path)

    def load_data(self, samples_to_load, data_path):
        # Data loader
        center_trajs, center_trajs_speeds, neighbors_, ped_ids_sorted, grids = [], [], [], [], []
        # Iterate over samples to loads
        for file_path in samples_to_load:
            if self.verbose:
                print(file_path)
            # Build temporary array holding neighbors for one sample
            neighbors_per_frame = [torch.Tensor() for x in range(20)]

            # Load center a pedestrian trajectory and its neighbors
            tracklet = pd.read_csv(
                data_path + file_path + '.csv',
                header=None, delimiter=',')
            center_traj = tracklet.iloc[:20, :]
            center_traj.loc[:, [2, 3]] = center_traj.loc[
                :, [2, 3]] * self.scaling_factor
            center_traj_torch = (
                en_cuda(torch.Tensor(center_traj.as_matrix())))
            map_frame_nb_to_idx = dict(
                [(x, idx) for idx, x in enumerate(center_traj[0].values)])

            neighbors = tracklet.iloc[20:,:]
            neighbors.loc[:, [2, 3]] = neighbors.loc[
                :, [2, 3]] * self.scaling_factor
            # Group by frame and fill neighbors_per_frame
            for frame_nb, x in neighbors.groupby(0):
                neighbors_per_frame[map_frame_nb_to_idx[frame_nb]] = en_cuda(
                    torch.Tensor(x.as_matrix()))
            # Create occupancy maps for the observed steps
            temp = []
            for idx, frame in enumerate(neighbors_per_frame[:8]):
                ggg = get_grid(frame, None, ped_data=center_traj_torch[idx, :][[2, 3]],
                               grid_size=self.grid_size, max_dist=self.max_dist_neighbors)
                temp.append(ggg.unsqueeze(0))

            # Append to global arrays
            temp = torch.cat(temp, 0)
            grids.append(temp)
            neighbors_.append(neighbors_per_frame)
            center_trajs.append(center_traj_torch.unsqueeze(0))
            center_trajs_speeds.append(
                compute_speeds(center_traj_torch).unsqueeze(0))
            ped_ids_sorted.append([])
        # Add to class
        self.center_trajs = center_trajs
        self.neighbors = neighbors_
        self.center_trajs_speeds = center_trajs_speeds
        self.ped_ids_sorted = ped_ids_sorted
        self.grids = torch.cat(grids, 1)
        #self.grids = self.grids.transpose(1, 0)

    def __iter__(self):
        # Return the iterable object (self)
        return self

    def next(self):
        # Index corresponding batch
        if self.index >= len(self.center_trajs):
            self.index = 0
            raise StopIteration
        start = self.index
        end = start + self.batch_size
        self.index = (self.index + self.batch_size)
        # Return corresponding batch
        return (self.center_trajs[start:end], self.center_trajs_speeds[start:end], self.neighbors[start:end], self.ped_ids_sorted[start:end], self.grids[:, start:end, :])

    def __len__(self):
        return len(self.center_trajs)

    def __next__(self):
        # For compatibility with Python3
        return self.next()
