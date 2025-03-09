import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  #选一个起点
        while timestep <= timestep_max and s != S[4]:
            timestep +=1
            #在状态s下选择动作
            rand = np.random.rand()
            temp = 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if rand < temp:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break

            rand = np.random.rand()
            temp = 0
            for s_opt in S:
                temp += P.get(join(join(s, a_opt), s_opt), 0)
                if rand < temp:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes


#采样5次,每个序列最长不超过20步
#第五条序列
#[('s4', '概率前往', 1, 's3'), ('s3', '前往s4', -2, 's4'), ('s4', '概率前往', 1, 's3'), ('s3', '前往s5', 0, 's5')]
# episodes = sample(MDP, Pi_1, 20, 5)
# print('第一条序列\n', episodes[0])
# print('第二条序列\n', episodes[1])
# print('第五条序列\n', episodes[4])

def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):
            (s, a, r, s_next) = episode[i]
            G = gamma * G + r
            N[s] = N[s] + 1
            V[s] = V[s] +(G - V[s])/N[s]

timestep_max = 20
# 采样1000次,可以自行修改
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)


