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
        self.epsilon_start = 50  # Episodes initial random policy is used
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
        if isinstance(p1, dict):
            p1 = [p1['x'], p1['y']]
        if isinstance(p2, dict):
            p2 = [p2['x'], p2['y']]
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

    # 获取p2到p1的连线上与p2一定距离的某个点
    def get_point_between_p1p2(p1, p2, dis):
        if isinstance(p1, dict):
            p1 = [p1['x'], p1['y']]
        if isinstance(p2, dict):
            p2 = [p2['x'], p2['y']]
        if p2[0] == p1[0]:
            y = p2[1] + dis if p2[1] < p1[1] else p2[1] - dis
            return [p2[0], y]
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        x1 = p2[0] + dis / math.sqrt(k ** 2 + 1)
        x2 = p2[0] - dis / math.sqrt(k ** 2 + 1)
        y1, y2 = k * (x1 - p1[0]) + p1[1], \
                 k * (x2 - p1[0]) + p1[1]

        if point2point([x1, y1], p1) < point2point([x2, y2], p1):
            return [x1, y1]
        return [x2, y2]

    # 由p1指向p2的向量角度，水平向右为0度，顺时针为正
    def get_angle(p1, p2):
        if isinstance(p1, dict):
            p1 = [p1['x'], p1['y']]
        if isinstance(p2, dict):
            p2 = [p2['x'], p2['y']]
        pp_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        angle = dot_product_angle(pp_vector, np.array([1, 0]))
        if pp_vector[1] < 0:
            angle = -angle
        return angle

    def pos_list2dict(pos_list):
        return {'x': pos_list[0], 'y': pos_list[1]}

    # TODO：奖励函数
    reward = 0
    right_goal_pos = [665, 400]
    left_goal_pos = [35, 400]

    agent_0_pos = pos_list2dict(attri_dict['pos'][0])
    agent_1_pos = pos_list2dict(attri_dict['pos'][1])
    ball_pos = pos_list2dict(attri_dict['pos'][2])
    ball_vel = attri_dict['v'][2]

    # agent_0的reward计算，agent_0['x']始终小于400
    dis2ball = point2point(ball_pos, agent_0_pos)
    if dis2ball < 60:
        n_step = 0.0
    elif dis2ball < 80:
        n_step = 0.1
    else:
        n_step = 0.3
    next_ball_pos = pos_list2dict([ball_pos['x'] + ball_vel[0] * n_step, ball_pos['y'] + ball_vel[1] * n_step])
    agent_angle = get_angle(left_goal_pos, agent_0_pos)  # 左侧球门指向agent_0的角度
    ball_angle = get_angle(left_goal_pos, ball_pos)
    agent2goal = point2point(agent_0_pos, left_goal_pos)
    ball2goal = point2point(left_goal_pos, ball_pos)

    if ball_pos['x'] < 350:  # 如果球在左半场，那么控球权就在agent_0手上
        target_pos = get_point_between_p1p2(left_goal_pos, next_ball_pos, 32)  # agent的目标位置应该是ball与goal连线上靠近ball的一点
        dis2target = point2point(agent_0_pos, target_pos)
        if agent2goal > ball2goal:  # 如果ball比agent更靠近球门，应该有一个惩罚
            reward -= agent2goal / 200
        elif dis2target > 120:
            reward -= dis2target / 800
        elif dis2target <= 120:
            reward += (120 - dis2target) / 30
    else:  # 如果球在右半场
        target_pos = get_point_between_p1p2(next_ball_pos, left_goal_pos, 300)
        dis2target = point2point(agent_0_pos, target_pos)
        reward += (200 - dis2target) / 200

    angle_error = abs(ball_angle - agent_angle)
    if agent2goal < ball2goal and dis2target <= 200 and angle_error <= 30:
        reward += (30 - angle_error) / 30

    return reward


# agent_0作为训练智能体，agent_1作为陪练智能体
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
            # try:
            #     env.env_core.render()
            # except:
            #     print('a')  # render可能会出错
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
