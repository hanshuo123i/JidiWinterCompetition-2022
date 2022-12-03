import torch
import os
import numpy as np
from agents.rule.submission import my_controller as controller_0
from agents.random.submission import my_controller as controller_1


def run(game, actions_space):
    env = game
    # 只保存player1的，也就是rule agent
    episode_state = []
    episode_action = []
    episode_done = []
    winners = []
    player0_models = []
    for i_ep in range(500_00):
        ep_timesteps = 0

        obs, attri_dict = env.reset()
        done = False
        error_flag = False

        while not done:
            try:
                # env.env_core.render()
                ep_timesteps += 1

                player0_obs = obs[0]
                player1_obs = obs[1]
                player0_act = controller_0(player0_obs, attri_dict, actions_space)
                player1_act = controller_1(player1_obs, attri_dict, actions_space)

                env_action = [player0_act, player1_act]
                next_obs, reward, done, _, _, next_attri_dict = env.step(env_action)

                episode_state.append(player0_obs['obs']['agent_obs'])
                episode_action.append(player0_act)
                episode_done.append(done)

                obs = next_obs
                attri_dict = next_attri_dict
            except:
                error_flag = True  # 代表该episode出错
                done = True
                episode_done[-1] = done

        winner = env.check_win() if not error_flag else '2'
        winners.append(winner)
        print(f'winner: {winner}, ep: {i_ep}')

        if (i_ep + 1) % 1000 == 0:
            save(episode_state, episode_action, episode_done, winners, i_ep)
            episode_state = []
            episode_action = []
            episode_done = []
            winners = []


def save(episode_state, episode_action, episode_done, winners, i_ep):
    file_name = os.path.dirname(os.path.abspath(__file__)) + '/rule_exp_data/' + 'ep_' + str(i_ep + 1) + '.npz'
    np.savez(file_name, obs=np.array(episode_state), act=np.array(episode_action).squeeze(),
             done=np.array(episode_done), winners=np.array(winners))
