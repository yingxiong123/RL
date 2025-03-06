import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    '''
        实现多臂老虎机的类定义
        属性：
            K:拉杆个数
            probs: k个拉杆中奖概率
            best_k:中奖概率最大的拉杆索引
            best_prob:中奖概率最大的拉杆中奖概率
    '''
    def __init__(self, K: int):
        self.K = K  
        self.probs = np.random.rand(K)
        self.best_k = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_k]

    def step(self,k):
        ''' 玩家选择了k步动作,得到的奖励 '''
        return 1 if np.random.rand() < self.probs[k] else 0 

K = 10
bandit = BernoulliBandit(10)

print(f'随机生成了{K}个拉杆，每个获奖概率为{bandit.probs}\n')
print(f'获奖概率最大的拉杆为{bandit.best_k}，概率是{bandit.best_prob}')

class Solver:
    ''' 实现对多臂老虎机的求解 '''
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit     #老虎机类
        self.regret = 0          #当前累计懊悔值
        self.regrets = []        #每步都记录，方便可视化
        self.actions = []        #记录动作
        self.counts = np.zeros(self.bandit.K)         #记录每一拉杆的执行次数，用来估计期望奖励

    def update_regrets(self, k: int):
        ''' 根据选的k拉杆,也就是action,计算当前步懊悔值 '''
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> int:
        ''' 输出当前步的action '''
        raise NotImplementedError

    def run(self, num_steps):
        ''' 执行一定次数,num_steps总步数 '''
        for _ in range(num_steps):
            k = self.run_one_step()
            self.actions.append(k)
            self.update_regrets(k)
            self.counts[k] +=1

class EpsilonGreedy(Solver):
    ''' 利用贪心算法进行选择动作，继承了Solver '''
    def __init__(self, bandit, epsilon=0.01):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimations = np.zeros(bandit.K)

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimations)
        #print(k)
        r = self.bandit.step(k)
        self.estimations[k] += 1. / (self.counts[k] + 1) * (r - self.estimations[k])
        return k

#----------------单个贪婪算法画图-------------------------------------------------
# def plot_results(solver):
#     plt.plot(solver.regrets)
#     plt.xlabel('Time steps')
#     plt.ylabel('Cumulative regrets')
#     plt.show()

# epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=0.01)
# epsilon_greedy_solver.run(5000)
# plot_results(epsilon_greedy_solver)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)

#----------------多个贪婪算法画图-------------------------------------------------
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


#----------------对比不同epsilon效果-------------------------------------------------
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [EpsilonGreedy(bandit, epsilon=e) for e in epsilons]
# epsilon_greedy_solver_names = [f"epsilon={e}" for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
#plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

#----------------上置信界算法-------------------------------------------------
class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(11)
coef = 0.01  # 控制不确定性比重的系数
UCB_solver = UCB(bandit, coef)
UCB_solver.run(500)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])


