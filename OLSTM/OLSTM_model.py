import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from OLSTM_utils import get_coef, sample_gaussian_2d, get_grid, en_cuda, rotate


class OLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(OLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.embedding_occupancy_map = args['embedding_occupancy_map']
        self.use_speeds = args['use_speeds']
        self.grid_size = args['grid_size']
        self.max_dist = args['max_dist']
        self.hidden_size = args['hidden_size']
        self.embedding_spatial = nn.Linear(2, self.embedded_input)
        self.embedding_o_map = nn.Linear(args['grid_size']**2, self.embedding_occupancy_map)
        self.lstm = nn.LSTM(self.embedded_input +
                            self.embedding_occupancy_map, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, input_data, grids, neighbors,first_positions, num=12):

        selector = en_cuda(torch.LongTensor([2, 3]))
        embedded_input = []
        grids = Variable(grids)
        obs = 8
        # Embedd input and occupancy maps
        for i in range(input_data.size()[0]):
            temp = input_data[i, :, :]
            embedded_input.append(
                F.relu(self.embedding_spatial(temp[:, selector])).unsqueeze(0))

        embedded_input = torch.cat(embedded_input, 0)
        embedded_o_map = [F.relu(self.embedding_o_map(grids[i, :, :])).unsqueeze(
            0) for i in range(grids.size()[0])]

        embedded_o_map = torch.cat(embedded_o_map, 0)
        inputs = torch.cat([embedded_input, embedded_o_map], 2)

        # Feed embedded in to lstm
        hidden = (autograd.Variable(en_cuda(torch.randn(1, inputs.size()[1], self.hidden_size))), autograd.Variable(
            en_cuda(torch.randn((1, inputs.size()[1], self.hidden_size)))))  # clean out hidden state
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(inputs[:-1, :, :], hidden)

        last = inputs[-1, :, :].unsqueeze(0)

        results = []
        # Generate outputs
        points = []
        position_tracker = None
        if self.use_speeds:
            position_tracker = first_positions.clone()

        for i in range(num):
            # get gaussian params for every point in batch
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(last, hidden)

            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            # Iterate over all batch points
            last_speeds = []
            last_grids = []
            temp_points = []
            for idx in range(res_params.size()[0]):

                mux = res_params.data[idx, 0]
                muy = res_params.data[idx, 1]
                sx = res_params.data[idx, 2]
                sy = res_params.data[idx, 3]
                rho = res_params.data[idx, 4]
                # Sample speeds
                speed = en_cuda(torch.Tensor(
                    sample_gaussian_2d(mux, muy, sx, sy, rho)))
                if self.use_speeds:
                    pts = torch.add(speed, position_tracker[idx, :])
                    position_tracker[idx, :] = pts.clone()
                else:
                    pts = speed
                temp_points.append(pts.unsqueeze(0))
                # Compute current position and get the occupancy map
                grid = Variable(get_grid(neighbors[idx][i+obs], None, ped_data= pts, max_dist=self.max_dist, grid_size=self.grid_size))
                last_speeds.append(speed.unsqueeze(0))
                last_grids.append(grid)

            # Compute embeddings
            last_speeds = torch.cat(last_speeds, 0)
            last_speeds = Variable(last_speeds)
            last_grids = torch.cat(last_grids, 0)
            last_speeds = F.relu(
                self.embedding_spatial(last_speeds)).unsqueeze(0)
            last_grids = F.relu(self.embedding_o_map(last_grids)).unsqueeze(0)
            last = torch.cat([last_speeds, last_grids], 2)
            points.append(torch.cat(temp_points, 0).unsqueeze(0))

        results = torch.cat(results, 0)
        points = torch.cat(points, 0)
        return results, points

    def get_hidden_states(self,input_,grid,hidden):
        pos = F.relu(
            self.embedding_spatial(input_)).unsqueeze(0)
        grids = F.relu(self.embedding_o_map(grid)).unsqueeze(0)
        lstm_input = torch.cat([pos, grids], 2)
        out, hidden = self.lstm(lstm_input, hidden)
        return hidden

