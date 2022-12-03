'''
    agent_1的规则，以下规则依据attri_dict实现，没法用于提交
'''
import numpy as np
import math


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
def get_point_between_p1p2(p1, p2, dis, is_inside=True):
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
        return [x1, y1] if is_inside else [x2, y2]
    return [x2, y2] if is_inside else [x1, y1]


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


def limit(value, min_value, max_value):
    return min(max_value, max(min_value, value))


class RuleAgent:
    def __init__(self, side='1'):
        self.side = side
        self.energy_recover_rate = 200  # 能量恢复速率
        self.goal_pos = [665, 400] if self.side == '1' else [35, 400]
        self.enemy_goal_pos = [35, 400] if self.side == '1' else [665, 400]

    def update(self, attri_dict):
        self.agent_pos = pos_list2dict(attri_dict['pos'][1]) if self.side == '1' else pos_list2dict(attri_dict['pos'][0])
        self.agent_vel = attri_dict['v'][1] if self.side == '1' else attri_dict['v'][0]
        self.agent_energy = attri_dict['energy'][1] if self.side == '1' else attri_dict['energy'][0]
        self.agent_theta = attri_dict['theta'][1][0] % 360 if self.side == '1' else attri_dict['theta'][0][0] % 360
        if self.agent_theta < 0:
            self.agent_theta = 360 + self.agent_theta
        self.ball_pos = pos_list2dict(attri_dict['pos'][2])
        self.ball_vel = attri_dict['v'][2]

        self.dis2ball = point2point(self.ball_pos, self.agent_pos)
        if self.dis2ball < 60:
            n_step = 0.0
        elif self.dis2ball < 80:
            n_step = 0.1
        else:
            n_step = 0.5
        self.next_ball_pos = pos_list2dict([self.ball_pos['x'] + self.ball_vel[0] * n_step,
                                            self.ball_pos['y'] + self.ball_vel[1] * n_step])
        self.agent_angle = get_angle(self.goal_pos, self.agent_pos)  # 右侧球门指向agent_0的角度
        self.ball_angle = get_angle(self.goal_pos, self.ball_pos)
        self.agent2goal = point2point(self.agent_pos, self.goal_pos)
        self.ball2goal = point2point(self.goal_pos, self.ball_pos)

    def get_force_action(self):
        v = math.sqrt(self.agent_vel[0] ** 2 + self.agent_vel[1] ** 2)
        if self.dis2ball < 60 and self.agent_pos['x'] > self.ball_pos['x']:  # 靠近ball且在ball的右侧时
            force = np.array([200])
            force = self.limit_energy_action(force, v, self.agent_energy)
        else:
            if self.agent_pos['x'] < self.ball_pos['x']:  # ball在agent的右侧时，agent应全力赶到ball的右侧
                force = np.array([200])
                force = self.limit_energy_action(force, v, self.agent_energy)
            else:  # 在ball的右侧，且离ball有一定距离
                force = np.array([limit(self.dis2ball * 2, -100, 200)])
                force = self.limit_energy_action(force, v, self.agent_energy)
        if self.ball_pos['x'] < 290:
            force = np.array([100])
            force = self.limit_energy_action(force, v, self.agent_energy)
        return force

    def get_angle_action(self):
        if self.agent_pos['x'] > self.ball_pos['x']:  # 在ball的右侧，agent应该赶到ball与球门连线上离ball近的某个点上去
            if self.ball_pos['x'] >= 350:  # ball在右半场
                if self.ball_pos['x'] <= 500:  # ball不咋靠近球门, agent应瞄准对面球门射击
                    target_pos = get_point_between_p1p2(self.enemy_goal_pos, self.next_ball_pos, 35, is_inside=False)
                else:
                    target_pos = get_point_between_p1p2(self.goal_pos, self.next_ball_pos, 32)
            else:
                target_pos = get_point_between_p1p2(self.next_ball_pos, self.goal_pos, 300)
        else:  # 在ball的左侧，agent应全力赶到ball的右侧，最好赶到球门前
            target_pos = get_point_between_p1p2(self.next_ball_pos, self.goal_pos, 50)
            target_pos[0] += 10

        self2target_angle = get_angle(self.agent_pos, target_pos)
        if self2target_angle < 0:
            self2target_angle = 360 + self2target_angle
        angle_error = self2target_angle - self.agent_theta
        if angle_error >= 180 or angle_error <= -180:
            if angle_error < 0:
                angle_error = 360 + angle_error
            else:
                angle_error = angle_error - 360
        angle = np.array([limit(angle_error, -30, 30)])
        return angle

    # 判断force是否过大导致耗能过多
    def limit_energy_action(self, force, v, my_energy, min_energy=100):
        consume_energy = (abs(force) * v / 50 - self.energy_recover_rate) * 0.1
        if my_energy - consume_energy > min_energy:
            pass
        else:
            force = ((my_energy - min_energy) * 10 + self.energy_recover_rate) * 50 / v
            force = np.array([limit(force, -100, 200)])
        return force


agent = RuleAgent()
def my_controller(observation, attri_dict, action_space, is_act_continuous=False):
    agent_action = [np.array([0.0]), np.array([0.0])]
    agent.update(attri_dict)
    agent_action[0] = agent.get_force_action()
    agent_action[1] = agent.get_angle_action()
    return agent_action
