import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from utils import get_coef, sample_gaussian_2d, en_cuda


class VLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(VLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.hidden_size = args['hidden_size']
        self.embedding = nn.Linear(2, self.embedded_input)
        self.lstm = nn.LSTM(self.embedded_input, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, x_data, num=12):
        # Embedd input
        embedded_input = [F.relu(self.embedding(x_data[i, :, :])).unsqueeze(
            0) for i in range(x_data.size()[0])]
        embedded_input = torch.cat(embedded_input, 0)
        inputs = embedded_input[:-1, :, :]

        # Feed embedded in to lstm
        hidden = (Variable(en_cuda(torch.randn(1, inputs.size()[1], self.hidden_size))), Variable(
            en_cuda(torch.randn((1, inputs.size()[1], self.hidden_size)))))  # clean out hidden state
        out, hidden = self.lstm(inputs, hidden)
        last = embedded_input[-1, :, :].unsqueeze(0)

        results = []
        # Generate outputs
        speeds = []
        for i in range(num):
            out, hidden = self.lstm(last, hidden)
            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            # Iterate over all batch points
            last = []
            temp_points = []
            for idx in range(res_params.size()[0]):
                mux = res_params.data[idx, 0]
                muy = res_params.data[idx, 1]
                sx = res_params.data[idx, 2]
                sy = res_params.data[idx, 3]
                rho = res_params.data[idx, 4]
                speed = en_cuda(torch.Tensor(
                    sample_gaussian_2d(mux, muy, sx, sy, rho)))
                last.append(speed.unsqueeze(0))
            last = torch.cat(last, 0)
            speeds.append(last.unsqueeze(0))
            last = Variable(last)
            last = F.relu(self.embedding(last)).unsqueeze(0)
        results = torch.cat(results, 0)
        speeds = (torch.cat(speeds, 0))
        return results, speeds

    def get_hidden_states(self, input_, hidden):
        pos = F.relu(self.embedding(input_)).unsqueeze(0)
        out, hidden = self.lstm(pos, hidden)
        return hidden
