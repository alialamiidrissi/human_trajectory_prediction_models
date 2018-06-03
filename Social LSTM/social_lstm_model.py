import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from model import VLSTM
from social_lstm_utils import *


class SLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(SLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.embedding_occupancy_map = args['embedding_occupancy_map']
        self.use_speeds = args['use_speeds']
        self.grid_size = args['grid_size']
        self.max_dist = args['max_dist']
        self.hidden_size = args['hidden_size']
        model_checkpoint = args['trained_model']
        self.trained_model = VLSTM(args)
        if model_checkpoint is not None:
            if torch.cuda.is_available():
                load_params = torch.load(model_checkpoint)
            else:
                load_params = torch.load(
                    model_checkpoint, map_location=lambda storage, loc: storage)
            self.trained_model.load_state_dict(load_params['state_dict'])

            # Freeze model
            for param in self.trained_model.parameters():
                param.requires_grad = False

        self.embedding_spatial = nn.Linear(2, self.embedded_input)
        self.embedding_o_map = nn.Linear(
            (args['grid_size']**2) * self.hidden_size,
            self.embedding_occupancy_map)
        self.lstm = nn.LSTM(self.embedded_input +
                            self.embedding_occupancy_map, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, input_data, grids, neighbors, first_positions, num=12):
        selector = en_cuda(torch.LongTensor([2, 3]))
        #grids = Variable(grids)
        obs = 8
        # Embedd input and occupancy maps
        # Iterate over frames
        hiddens_neighb = [(autograd.Variable(en_cuda(torch.zeros(1, 1, self.hidden_size))), autograd.Variable(
            en_cuda(torch.zeros((1, 1, self.hidden_size))))) for i in range(len(neighbors[0]) + 1)]
        inputs = None
        temp = None
        for i in range(input_data.size()[0]):
            temp = input_data[i, :].unsqueeze(0)
            embedded_input = F.relu(self.embedding_spatial(
                temp[:, selector])).unsqueeze(0)
            # Iterate over peds if there is neighbors
            social_tensor = None

            # Check if the pedestrians has neighbors
            if len(neighbors[0]):
                all_frame = torch.cat(
                    [input_data[i, :].data.unsqueeze(0), neighbors[i]], 0)
                valid_indexes_center = grids[i][1]
                # Iterate over valid neighbors if exists

                if valid_indexes_center is not None:
                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes_center:
                        # Get hidden states from VLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    if i>0:
                        last_obs = neighbors[i - 1][valid_indexes_center, :]
                        vlstm_in = neighbors[i][valid_indexes_center, :][:,[
                                    2, 3]] - last_obs[:,[2, 3]]
                        select = (last_obs[:,1] == -1).nonzero()
                        if len(select.size()):
                            vlstm_in[select.squeeze(1)] = 0

                    else:
                        vlstm_in = en_cuda(torch.zeros(len(valid_indexes_center),2))



                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.cat(buff_hidden_c,1),torch.cat(buff_hidden_h,1)))

                    for idx,k in enumerate(valid_indexes_center):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes_center]

                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=grids[i][0], grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            embedded_o_map = F.relu(
                self.embedding_o_map(social_tensor)).unsqueeze(0)
            inputs = torch.cat([embedded_input, embedded_o_map], 2)
            if i == (obs - 1):
                break
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

        # Predict

        last = inputs

        results = []
        # Generate outputs
        points = []
        first_positions_c = first_positions.clone()
        for i in range(num):
            # get gaussian params for every point in batch
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            last_speeds = []
            last_grids = []
            temp_points = []
            mux = res_params.data[0, 0]
            muy = res_params.data[0, 1]
            sx = res_params.data[0, 2]
            sy = res_params.data[0, 3]
            rho = res_params.data[0, 4]
            # Sample speeds
            speed = en_cuda(torch.Tensor(
                sample_gaussian_2d(mux, muy, sx, sy, rho)))
            pts = torch.add(speed, first_positions_c[0, :])
            first_positions_c[0, :] = pts

            # Compute embeddings  and social grid

            last_speeds = speed.unsqueeze(0)
            last_speeds = Variable(last_speeds)
            pts_frame = pts.unsqueeze(0)

            # SOCIAL TENSOR
            # Check if neighbors exists
            if(len(neighbors[i + obs])):
                pts_w_metadata = en_cuda(torch.Tensor(
                    [[neighbors[i + obs][0, 0], input_data[0, 1].data[0], pts[0], pts[1]]]))
                frame_all = torch.cat(
                    [pts_w_metadata, neighbors[i + obs]], 0)
                # Get positions in social_grid
                (indexes_in_grid, valid_indexes) = get_grid_positions(neighbors[i + obs], None, ped_data=pts_frame.squeeze(0),
                                                                      grid_size=self.grid_size, max_dist=self.max_dist)
                if(valid_indexes is not None):

                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes:
                        # Get hidden states from OLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    last_obs = neighbors[i + obs - 1][valid_indexes, :]
                    vlstm_in = neighbors[i+ obs][valid_indexes, :][:,[
                                2, 3]] - last_obs[:,[2, 3]]
                    select = (last_obs[:,1] == -1).nonzero()
                    if len(select.size()):
                        vlstm_in[select.squeeze(1)] = 0


                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.cat(buff_hidden_c,1),torch.cat(buff_hidden_h,1)))

                    for idx,k in enumerate(valid_indexes):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    # Compute social tensor
                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes]
                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=indexes_in_grid, grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            last_speeds = F.relu(
                self.embedding_spatial(last_speeds)).unsqueeze(0)
            last_grids = F.relu(self.embedding_o_map(
                social_tensor)).unsqueeze(0)
            last = torch.cat([last_speeds, last_grids], 2)
            points.append(pts_frame.unsqueeze(0))

        results = torch.cat(results, 0)
        points = torch.cat(points, 0)
        return results, points
