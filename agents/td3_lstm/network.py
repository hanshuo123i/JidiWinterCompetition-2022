import torch
import math
from torch import nn
from torch.nn import functional as F


class FixupResNetCNN(nn.Module):
    """source: https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py"""

    class _FixupResidual(nn.Module):
        def __init__(self, depth, num_residual, reduction=16):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for p in self.conv1.parameters():
                p.data.mul_(1 / math.sqrt(num_residual))
            for p in self.conv2.parameters():
                p.data.zero_()
            self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

        def forward(self, x):
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4

            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        depth_in = input_channels

        layers = []
        if not double_channels:
            channel_sizes = [16, 32, 16]
        else:
            channel_sizes = [32, 64, 32]
        for depth_out in channel_sizes:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._FixupResidual(depth_out, 8),
                self._FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            self._FixupResidual(depth_in, 8),
            self._FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(40 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)


class Mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128,
                 num_layers=2, activation=nn.ReLU, last_activation=True):
        super().__init__()
        if type(hidden_size) == int:
            hidden_sizes = [hidden_size] * num_layers
        else:
            hidden_sizes = hidden_size
            assert len(hidden_sizes) == num_layers
        hidden_sizes.append(output_size)

        in_size = input_size
        layers = []
        for idx in range(len(hidden_sizes)):
            out_size = hidden_sizes[idx]
            if idx == len(hidden_sizes) - 1 and not last_activation:
                layers.extend([
                    nn.Linear(in_size, out_size)
                ])
            else:
                layers.extend([
                    nn.Linear(in_size, out_size),
                    activation(),
                ])
            in_size = out_size
        self.mlp = nn.Sequential(*layers)
        self.output_size = out_size

    def forward(self, x):
        return self.mlp(x)


class PolicyCNNLSTM(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super(PolicyCNNLSTM, self).__init__()

        self.cnn = FixupResNetCNN(input_channels=1)
        # self.mlp = Mlp(act_dim, self.cnn.output_size, num_layers=0)

        # fc branch
        self.mlp1 = Mlp(self.cnn.output_size, hidden_dim, num_layers=1)
        # lstm branch
        self.mlp2 = Mlp(self.cnn.output_size, hidden_dim, num_layers=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # contact
        self.mlp3 = Mlp(2 * hidden_dim, act_dim, num_layers=1, last_activation=False)

    def forward(self, obs, hidden_in=None):
        if obs.ndim == 5:
            _obs = obs.contiguous().view(-1, obs.size()[-3], obs.size()[-2], obs.size()[-1])
        else:
            _obs = obs
        img_feature = self.cnn(_obs)
        img_feature = img_feature.contiguous().view(-1, self.cnn.output_size)
        img_feature = img_feature.reshape(obs.size()[0], obs.size()[1], -1)

        # # last_act_feature = self.mlp(last_act)
        # if last_act_feature.ndim == 2:
        #     last_act_feature = last_act_feature.unsqueeze(dim=0)

        fc_branch = self.mlp1(img_feature)

        lstm_branch = self.mlp2(img_feature)
        if hidden_in is not None:
            lstm_branch, lstm_hidden = self.lstm(lstm_branch, hidden_in)
        else:
            lstm_branch, lstm_hidden = self.lstm(lstm_branch)

        cat_branch = torch.cat([fc_branch, lstm_branch], -1)
        act = F.tanh(self.mlp3(cat_branch))

        return act, lstm_hidden


class QNetCNNLSTM(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super(QNetCNNLSTM, self).__init__()

        self.cnn = FixupResNetCNN(input_channels=1)
        self.act_mlp = Mlp(act_dim, self.cnn.output_size, num_layers=0)
        self.last_act_mlp = Mlp(act_dim, self.cnn.output_size, num_layers=0)

        # fc branch
        self.mlp1 = Mlp(2 * self.cnn.output_size, hidden_dim, num_layers=1)
        # lstm branch
        self.mlp2 = Mlp(self.cnn.output_size, hidden_dim, num_layers=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # contact
        self.mlp3 = Mlp(2 * hidden_dim, 1, num_layers=1, last_activation=False)

    def forward(self, obs, act, hidden_in=None):
        if obs.ndim == 5:
            _obs = obs.contiguous().view(-1, obs.size()[-3], obs.size()[-2], obs.size()[-1])
        else:
            _obs = obs
        img_feature = self.cnn(_obs)
        img_feature = img_feature.contiguous().view(-1, self.cnn.output_size)
        img_feature = img_feature.reshape(obs.size()[0], obs.size()[1], -1)
        act_feature = self.act_mlp(act)

        fc_branch = self.mlp1(torch.cat([img_feature, act_feature], -1))

        lstm_branch = self.mlp2(img_feature)
        if hidden_in is not None:
            lstm_branch, lstm_hidden = self.lstm(lstm_branch, hidden_in)
        else:
            lstm_branch, lstm_hidden = self.lstm(lstm_branch)

        cat = torch.cat([fc_branch, lstm_branch], -1)

        q_value = self.mlp3(cat)

        return q_value, lstm_hidden
