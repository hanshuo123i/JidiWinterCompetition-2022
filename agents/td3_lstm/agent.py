import torch.optim as optim
from torch.nn.utils import rnn
import os
import numpy as np
from .network import *


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
        self.q_net1.load_state_dict(torch.load(path + '/model/_q1_' + str(epi) + '.pt', map_location=self.device))
        self.q_net2.load_state_dict(torch.load(path + '/model/_q2_' + str(epi) + '.pt', map_location=self.device))
        self.policy_net.load_state_dict(torch.load(path + '/model/_policy_' + str(epi) + '.pt', map_location=self.device))
        self.q_net1.train()
        self.q_net2.train()
        self.policy_net.train()


def pack_batch(batch, device):
    packed_sequence = rnn.pad_sequence([torch.FloatTensor(np.array(i)).to(device) for i in batch], batch_first=True)

    return packed_sequence