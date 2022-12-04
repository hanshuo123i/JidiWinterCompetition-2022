import random
import datetime
import copy
import numpy as np
from network import *
from tensorboardX import SummaryWriter
import torch
from torch import nn


def process_obs(obs):
    obs = copy.deepcopy(obs)
    if obs[32][19] == 8:  # player1的观察
        obs[obs == 10] = -10  # 敌方位置替换为-10
        obs[obs == 8] = 10  # 将自身位置替换为10
    else:  # player2的观察
        obs[obs == 8] = -10  # 自身位置已经为10，将敌方位置变为-10

    return obs


eps_data = np.load('ep_1000.npz')

obs_data = eps_data['obs']
act_data = eps_data['act']
done = eps_data['done']
winners = eps_data['winners']
num_episodes = len(winners)
epi_end_idx = np.where(done==True)[0]
epi_start_idx = epi_end_idx[:-1] + 1
epi_start_idx = np.insert(epi_start_idx, 0, 0)


long_episode = []
for i in range(num_episodes):
    if epi_end_idx[i] - epi_start_idx[i] > 100:
        long_episode.append(i)

device = torch.device("cuda")
torch.cuda.set_device(0)
train_step = 100000
policy = PolicyCNNLSTM(40 * 40, 256, 2).to(device)
loss_func = nn.MSELoss()
optim = torch.optim.Adam(policy.parameters(), lr=0.001)


cur_time = datetime.datetime.now()
writer = SummaryWriter(f'./log/bc_loss_{cur_time.day}_{cur_time.hour}_{cur_time.minute}')


processed_obs_data = []
for obs in obs_data:
    processed_obs_data.append(process_obs(obs))
processed_obs_data = np.array(processed_obs_data)


for t_step in range(train_step):
    while True:
        if random.random() > 0.7:
            sample_epi_idx = random.randint(0, num_episodes - 1)
        else:
            sample_epi_idx = random.sample(long_episode, 1)[0]
        if winners[sample_epi_idx] in ['1', '-1']:
            break
    sample_obs = processed_obs_data[epi_start_idx[sample_epi_idx]: epi_end_idx[sample_epi_idx] + 1]
    sample_act = act_data[epi_start_idx[sample_epi_idx]: epi_end_idx[sample_epi_idx] + 1]

    sample_obs_tensor = torch.FloatTensor(sample_obs).unsqueeze(dim=1).to(device)
    sample_act_tensor = torch.FloatTensor(sample_act).unsqueeze(dim=1).to(device)

    predict_act, _ = policy(sample_obs_tensor)
    loss = loss_func(predict_act, sample_act_tensor)

    optim.zero_grad()
    loss.backward()
    optim.step()

    writer.add_scalar('bc_loss_rule', loss, t_step)
    if t_step % 100 == 0:
        print("loss:", loss.cpu().data.numpy(), "t_step:", t_step)
        torch.save(policy.state_dict(), 'bc_rule_ACTCNNLSTM.pt')
