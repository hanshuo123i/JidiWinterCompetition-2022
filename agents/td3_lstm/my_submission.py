import copy
import math
from re import L
import torch
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


class TD3Config:
    def __init__(self) -> None:
        self.algo = 'TD3'  # 算法名称
        self.device = torch.device("cuda")  # 检测GPU
        self.train_eps = 8000  # 训练的回合数
        self.start_timestep = 25e3  # Time steps initial random policy is used
        self.epsilon_start = -1  # Episodes initial random policy is used
        self.eval_freq = 5  # How often (episodes) we evaluate
        self.max_timestep = 100000  # Max time steps to run environment
        self.expl_noise = 0.3  # Std of Gaussian exploration noise
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

        self.is_test = False
        self.if_my_rew = True


# 非对称对抗似乎不需要
def process_obs(obs):
    obs = copy.deepcopy(obs)
    if obs[32][19] == 8:  # player1的观察
        obs[obs == 10] = -10  # 敌方位置替换为-10
        obs[obs == 8] = 10  # 将自身位置替换为10
    else:  # player2的观察
        obs[obs == 8] = -10  # 自身位置已经为10，将敌方位置变为-10
    return obs


from .buffer import *
from .agent import TD3_LSTM


def get_reward(attri_dict, action):
    def point2point(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def dot_product_angle(v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            print("Zero magnitude vector!")
        else:
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            return angle
        return 0

    # 获取敌方到center的连线上与enemy一定距离的某个点
    def get_center_enemy_point(enemy_pos, dis):
        center_pos = [350, 400]  # 场地中心
        if enemy_pos[0] == center_pos[0]:
            y = enemy_pos[1] + dis if enemy_pos[1] < center_pos[1] else enemy_pos[1] - dis
            return [enemy_pos[0], y]
        k = (enemy_pos[1] - center_pos[1]) / (enemy_pos[0] - center_pos[0])
        x1 = enemy_pos[0] + dis / math.sqrt(k ** 2 + 1)
        x2 = enemy_pos[0] - dis / math.sqrt(k ** 2 + 1)
        y1, y2 = k * (x1 - center_pos[0]) + center_pos[1], \
                 k * (x2 - center_pos[0]) + center_pos[1]

        if point2point([x1, y1], center_pos) < point2point([x2, y2], center_pos):
            return [x1, y1]
        return [x2, y2]

    # 由p1指向p2的向量角度，水平向右为0度，时针为正
    def get_angle(p1, p2):
        pp_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        angle = dot_product_angle(pp_vector, np.array([1, 0]))
        if pp_vector[1] < 0:
            angle = -angle
        return angle

    # TODO：奖励函数
    reward = 0

    return reward


def train(cfg, env, agent, enemy_controller, action_space, replay_buffer):
    print('开始训练!')
    rewards = []  # 记录所有回合的奖励
    if cfg.is_test:
        print('加载模型')
        agent.load(os.path.dirname(os.path.abspath(__file__)), 3499)
    else:
        cur_time = datetime.datetime.now()
        writer = SummaryWriter(f'./log/{cur_time.day}_{cur_time.hour}_{cur_time.minute}')

    for i_ep in range(int(cfg.train_eps)):
        ep_reward = 0
        ep_timesteps = 0
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []

        obs, attri_dict = env.reset()
        done = False
        agent_obs = process_obs(obs[0]['obs']['agent_obs'])
        agent_obs = np.expand_dims(agent_obs, axis=0)
        enemy_obs = obs[1]

        hidden_out = (torch.zeros([1, 1, cfg.hidden_dim], dtype=torch.float).to(cfg.device),
                      torch.zeros([1, 1, cfg.hidden_dim], dtype=torch.float).to(cfg.device))
        agent_last_action = np.array([np.random.random() * 2 - 1, np.random.random() * 2 - 1])

        while not done:
            hidden_in = hidden_out
            try:
                env.env_core.render()
            except:
                print('a')  # render可能会出错
            ep_timesteps += 1

            if cfg.is_test:
                agent_action, hidden_out = agent.choose_original_action(np.array(agent_obs), hidden_in)
            else:
                if i_ep < cfg.epsilon_start:
                    agent_action = np.array([np.random.random() * 2 - 1, np.random.random() * 2 - 1])
                else:
                    agent_action, hidden_out = agent.choose_original_action(np.array(agent_obs), hidden_in)
                    agent_action = (agent_action + np.random.normal(0, cfg.expl_noise, size=cfg.n_actions)).clip(-1, 1)

            # 转为环境动作
            agent_env_action = [np.array([agent_action[0] * 150 + 50]), np.array([agent_action[1] * 30])]
            # 选择敌方动作
            enemy_env_action = enemy_controller(enemy_obs, attri_dict, action_space[1][0], is_act_continuous=True)

            env_action = [agent_env_action, enemy_env_action]
            next_obs, reward, done, _, _, next_attri_dict = env.step(env_action)

            if cfg.if_my_rew:
                my_reward = get_reward(next_attri_dict, agent_env_action)
            else:
                my_reward = 0

            next_agent_obs = process_obs(next_obs[0]['obs']['agent_obs'])
            next_agent_obs = np.expand_dims(next_agent_obs, axis=0)
            next_enemy_obs = next_obs[1]
            if reward[0] > reward[1]:
                reward = 15
            elif reward[0] == reward[1]:
                reward = 0
            else:
                reward = -15
            agent_reward = reward + my_reward

            if ep_timesteps == 1:
                init_hidden_in = hidden_in
                init_hidden_out = hidden_out
            episode_state.append(agent_obs)
            episode_action.append(agent_action)
            episode_last_action.append(agent_last_action)
            episode_reward.append([agent_reward])
            episode_next_state.append(next_agent_obs)
            episode_done.append([done])

            agent_obs = next_agent_obs
            agent_last_action = agent_action
            enemy_obs = next_enemy_obs
            attri_dict = next_attri_dict
            ep_reward += agent_reward

        if i_ep > 10 and i_ep + 1 >= cfg.epsilon_start and not cfg.is_test:
            print('训练100次...')
            for _ in range(100):
                agent.update()
            print('训练完成！')

        replay_buffer.push(init_hidden_in, init_hidden_out, episode_state, episode_action,
                                      episode_last_action, episode_reward, episode_next_state, episode_done)

        if (i_ep + 1) % 1 == 0:
            print('回合：{}/{}, 奖励：{:.2f}, 步数：{}, winner：{}'.
                  format(i_ep + 1, cfg.train_eps, ep_reward / ep_timesteps, ep_timesteps, env.check_win()))
        if not cfg.is_test:
            writer.add_scalar('rewards_td3lstm', ep_reward / ep_timesteps, i_ep)
            writer.add_scalar('winner_td3lstm', int(env.check_win()), i_ep)
            if (i_ep + 1) % 100 == 0:
                print('保存模型')
                curr_path = os.path.dirname(os.path.abspath(__file__))
                agent.save(os.path.join(curr_path, 'model'), i_ep)

        rewards.append(ep_reward / ep_timesteps)

    print('完成训练！')
    return rewards


def run(game, action_space, enemy_controller):
    torch.cuda.set_device(0)
    cfg = TD3Config()
    replay_buffer = ReplayBufferLSTM(cfg.replay_buffer_size)
    agent = TD3_LSTM(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg, replay_buffer)
    train(cfg, game, agent, enemy_controller, action_space, replay_buffer)
