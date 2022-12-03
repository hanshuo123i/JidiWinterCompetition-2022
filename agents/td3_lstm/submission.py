'''
    此文件用于提交
'''


import torch.optim as optim
from torch.nn.utils import rnn
import math
import os
import copy
from torch import nn
from torch.nn import functional as F
import random
import torch
import numpy as np


class TD3Config:
    def __init__(self) -> None:
        self.algo = 'TD3'  # 算法名称
        self.device = torch.device("cpu")  # 检测GPU
        self.train_eps = 8000  # 训练的回合数
        self.start_timestep = 25e3  # Time steps initial random policy is used
        self.epsilon_start = 50  # Episodes initial random policy is used
        self.eval_freq = 5  # How often (episodes) we evaluate
        self.max_timestep = 100000  # Max time steps to run environment
        self.expl_noise = 0.2  # Std of Gaussian exploration noise
        self.gamma = 0.97  # gamma factor
        self.tau = 0.005  # 软更新
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 5  # Frequency of delayed policy updates

        self.n_states = 1600  # 状态维度
        self.n_actions = 2  # 动作维度
        self.hidden_dim = 256

        self.batch_size = 2  # each sample contains an episode for lstm policy
        self.replay_buffer_size = 5e5

        self.is_test = True
        self.load_model_epi = 2999
        self.if_my_rew = True


def process_obs(obs):
    obs = copy.deepcopy(obs)
    if obs[32][19] == 8:  # player1的观察
        obs[obs == 10] = -10  # 敌方位置替换为-10
        obs[obs == 8] = 10  # 将自身位置替换为10
    else:  # player2的观察
        obs[obs == 8] = -10  # 自身位置已经为10，将敌方位置变为-10
    return obs


class ReplayBufferLSTM:
    """
    Replay buffer for agent with LSTM network additionally storing previous action,
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


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


class TD3_LSTM:
    def __init__(self, obs_dim, hidden_dim, act_dim, cfg, replay_buffer):
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.update_cnt = 0

        self.replay_buffer = replay_buffer

        self.q_net1 = QNetCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)
        self.q_net2 = QNetCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)
        self.target_q_net1 = QNetCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)
        self.target_q_net2 = QNetCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)
        self.policy_net = PolicyCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)
        # self.policy_net.load_state_dict(torch.load('agents/td3_lstm/model/bc_rule_CNNLSTM.pt', map_location=self.device))
        self.target_policy_net = PolicyCNNLSTM(obs_dim, hidden_dim, act_dim).to(self.device)

        self.target_q_net1 = self.target_init(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_init(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_init(self.policy_net, self.target_policy_net)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=3e-4)

    def target_init(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        return target_net

    def choose_original_action(self, obs, hidden_in, noise_scale=1.0):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0).to(self.device)

        act, hidden_out = self.policy_net(obs, hidden_in)
        return act.cpu().data.numpy().flatten(), hidden_out

    def update(self):
        self.update_cnt += 1

        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        state = pack_batch(state, self.device)
        next_state = pack_batch(next_state, self.device)
        action = pack_batch(action, self.device)
        last_action = pack_batch(last_action, self.device)
        reward = pack_batch(reward, self.device)
        done = pack_batch(done, self.device)

        # Q-Net Training
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action, _ = self.target_policy_net(next_state)
            next_action = next_action + noise

            # Compute the target Q value
            target_Q1, _ = self.target_q_net1(next_state, next_action, hidden_out)
            target_Q2, _ = self.target_q_net2(next_state, next_action, hidden_out)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        q_value1, _ = self.q_net1(state, action)
        q_value2, _ = self.q_net2(state, action)

        q_value_loss1 = ((q_value1 - target_Q.detach()) ** 2).mean()
        q_value_loss2 = ((q_value2 - target_Q.detach()) ** 2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        if self.update_cnt % self.policy_freq == 0:
            new_action, _ = self.policy_net(state)

            new_q_value, _ = self.q_net1(state, new_action)

            policy_loss = - new_q_value.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net)

        self.update_cnt += 1

    def save(self, path, epi):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.q_net1.state_dict(), path + '/_q1_' + str(epi) + '.pt')
        torch.save(self.q_net2.state_dict(), path + '/_q2_' + str(epi) + '.pt')
        torch.save(self.policy_net.state_dict(), path + '/_policy_' + str(epi) + '.pt')

    def load(self, path, epi):
        print(f'加载{path}处模型！')
        self.q_net1.load_state_dict(torch.load(path + '/_q1_' + str(epi) + '.pt', map_location=self.device))
        self.q_net2.load_state_dict(torch.load(path + '/_q2_' + str(epi) + '.pt', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_' + str(epi) + '.pt', map_location=self.device))
        self.q_net1.train()
        self.q_net2.train()
        self.policy_net.train()


def pack_batch(batch, device):
    packed_sequence = rnn.pad_sequence([torch.FloatTensor(np.array(i)).to(device) for i in batch], batch_first=True)

    return packed_sequence


cfg = TD3Config()
replay_buffer = ReplayBufferLSTM(cfg.replay_buffer_size)
agent = TD3_LSTM(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg, copy.deepcopy(replay_buffer))
agent.load(os.path.dirname(os.path.abspath(__file__)), cfg.load_model_epi)
hidden_out = (torch.zeros([1, 1, cfg.hidden_dim], dtype=torch.float).to(cfg.device),
              torch.zeros([1, 1, cfg.hidden_dim], dtype=torch.float).to(cfg.device))


def my_controller(observation, action_space, is_act_continuous=False):
    global hidden_out
    hidden_in = hidden_out
    agent_obs = process_obs(observation['obs']['agent_obs'])
    agent_obs = np.expand_dims(agent_obs, axis=0)
    agent_action, hidden_out = agent.choose_original_action(np.array(agent_obs), hidden_in)
    agent_env_action = [np.array([agent_action[0] * 150 + 50]), np.array([agent_action[1] * 30])]
    return agent_env_action

