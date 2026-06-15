# -*- coding: utf-8 -*-
"""
训练主程序，让两个智能体通过试错学习
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from agent import QLearningAgent

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#检查位置是否被占用
def is_position_blocked(pos,agent_a,agent_b,env):
    """
    返回True的情况：
    超出地图边界
    是障碍物
    被另一个智能体占据（碰撞）
    """
    x,y=pos
    #检查边界
    if not env.is_valid(pos):
        return True
    #检查障碍物
    if env.is_obstacle(pos):
        return True
    #检查是否碰撞
    if pos==(agent_a.x,agent_a.y) or pos==(agent_b.x,agent_b.y):
        return True
    return False

#执行两个智能体的动作，处理碰撞，计算奖励
def execute_actions(agent_a, agent_b, env, action_a, action_b):
    """
    返回：
    new_pos_a, new_pos_b: 新位置
    reward_a, reward_b: 奖励
    done: 是否结束
    """
    # 检查是否已经到达终点
    a_at_goal = agent_a.has_reached_goal()
    b_at_goal = agent_b.has_reached_goal()
    # 如果两个都到达终点，直接结束
    if a_at_goal and b_at_goal:
        return (agent_a.x, agent_a.y), (agent_b.x, agent_b.y), 0, 0, True
    # 计算各自想去的下一个位置（已在终点的不能移动）
    if a_at_goal:
        next_a_pos = (agent_a.x, agent_a.y)
        action_a = 4  # 强制停留
    else:
        next_a_pos = agent_a.get_next_position(action_a)
    if b_at_goal:
        next_b_pos = (agent_b.x, agent_b.y)
        action_b = 4  # 强制停留
    else:
        next_b_pos = agent_b.get_next_position(action_b)
    a_reached = False
    b_reached = False
    # 情况1：两机想移动到同一个格子（且都没在终点）
    if next_a_pos == next_b_pos and not (a_at_goal or b_at_goal):
        reward_a = -5.0
        reward_b = -5.0
        next_a_pos = (agent_a.x, agent_a.y)
        next_b_pos = (agent_b.x, agent_b.y)
        agent_a.collisions += 1
        agent_b.collisions += 1
        return next_a_pos, next_b_pos, reward_a, reward_b, False
    # 初始奖励（已在终点的奖励为0）
    reward_a = 0 if a_at_goal else -0.1
    reward_b = 0 if b_at_goal else -0.1
    # 处理A的移动
    if not a_at_goal:
        a_blocked = is_position_blocked(next_a_pos, agent_a, agent_b, env)
        if a_blocked:
            next_a_pos = (agent_a.x, agent_a.y)
            reward_a = -5.0
            agent_a.collisions += 1
        else:
            if next_a_pos == agent_a.goal:
                a_reached = True
                reward_a = 10.0
    else:
        next_a_pos = (agent_a.x, agent_a.y)
    # 处理B的移动
    if not b_at_goal:
        b_blocked = is_position_blocked(next_b_pos, agent_a, agent_b, env)
        if b_blocked:
            next_b_pos = (agent_b.x, agent_b.y)
            reward_b = -5.0
            agent_b.collisions += 1
        else:
            if next_b_pos == agent_b.goal:
                b_reached = True
                reward_b = 10.0
    else:
        next_b_pos = (agent_b.x, agent_b.y)
    # 检查移动后是否相撞（且不在终点）
    if next_a_pos == next_b_pos and next_a_pos not in [agent_a.goal, agent_b.goal]:
        reward_a = -5.0
        reward_b = -5.0
        agent_a.collisions += 1
        agent_b.collisions += 1
        next_a_pos = (agent_a.x, agent_a.y)
        next_b_pos = (agent_b.x, agent_b.y)
    # 判断是否都到达终点
    done = (a_reached or a_at_goal) and (b_reached or b_at_goal)
    return next_a_pos, next_b_pos, reward_a, reward_b, done

#训练主函数
def train(episodes=500,verbose=True):
    """
    参数：
    episodes:训练轮数
    verbose:是否打印进度
    """
    env=Environment()
    agent_a=QLearningAgent(env, 'A',learning_rate=0.1,discount=0.9,epsilon=0.2)
    agent_b=QLearningAgent(env, 'B',learning_rate=0.1,discount=0.9,epsilon=0.2)
    #记录训练数据
    episode_rewards=[]
    episode_successes=[]
    episode_steps=[]
    print("开始训练……")
    print("="*50)
    for episode in range(episodes):
        #重置环境
        agent_a.reset()
        agent_b.reset()
        total_reward=0 
        max_steps=200 
        done=False
        for step in range(max_steps):
            #每个智能体选择动作
            action_a=agent_a.choose_action(training=True)
            action_b=agent_b.choose_action(training=True)
            #记录旧状态
            old_state_a=agent_a.get_state()
            old_state_b=agent_b.get_state()
            #执行动作，得到新位置和奖励
            new_a_pos,new_b_pos,reward_a,reward_b,done=execute_actions(agent_a, agent_b, env, action_a, action_b)
            #更新位置
            agent_a.set_position(*new_a_pos)
            agent_b.set_position(*new_b_pos)
            #更新Q表
            agent_a.update_Q(old_state_a, action_a, reward_a, agent_a.get_state(), done)
            agent_b.update_Q(old_state_b, action_b, reward_b, agent_b.get_state(), done)
            #累计奖励
            total_reward+=reward_a+reward_b
            if done:
                break
        episode_rewards.append(total_reward)
        episode_successes.append(1 if done else 0)
        episode_steps.append(step+1)
        #打印进度
        if verbose and (episode+1)%50==0:
            recent_success_rate=np.mean(episode_successes[-50:])*100
            recent_avg_reward=np.mean(episode_rewards[-50:])
            recent_avg_steps=np.mean(episode_steps[-50:])
            print(f"轮数：{episode+1:3d}/{episodes}|"
                  f"成功率：{recent_success_rate:.1f}%|"
                  f"平均奖励：{recent_avg_reward:.2f}|"
                  f"平均步数：{recent_avg_steps:.1f}")
    print("="*50)
    print("训练完成！")
    #保存训练好的Q表
    np.save('q_table_a.npy',agent_a.Q_table)
    np.save('q_table_b.npy',agent_b.Q_table)
    print("Q表已保存为q_table_a.npy和q_table_b.npy")
    return agent_a,agent_b,episode_rewards,episode_successes,episode_steps

#绘制训练曲线
def plot_training_results(rewards,successes,steps):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    #奖励曲线
    axes[0].plot(rewards,alpha=0.5,label='原始奖励')
    #添加平滑曲线
    window=20
    smooth_rewards=np.convolve(rewards, np.ones(window)/window,mode='valid')
    axes[0].plot(range(window-1,len(rewards)),smooth_rewards,'r-',linewidth=2,label='平滑曲线')
    axes[0].set_xlabel('训练轮数')
    axes[0].set_ylabel('总奖励')
    axes[0].set_title('学习曲线')
    axes[0].legend()
    axes[0].grid(True,alpha=0.3)
    #成功率曲线
    window=50 
    success_rate=np.convolve(successes, np.ones(window)/window,mode='valid')
    axes[1].plot(range(window-1,len(successes)),success_rate*100,'g-',linewidth=2)
    axes[1].set_xlabel('训练轮数')
    axes[1].set_ylabel('成功率（%）')
    axes[1].set_title('成功率曲线')
    axes[1].set_ylim(0,100)
    axes[1].grid(True,alpha=0.3)
    #步数曲线
    window=20
    smooth_steps=np.convolve(steps, np.ones(window)/window,mode='valid')
    axes[2].plot(range(window-1,len(steps)),smooth_steps,'b-',linewidth=2)
    axes[2].set_xlabel('训练轮数')
    axes[2].set_ylabel('完成步数')
    axes[2].set_title('收敛速度')
    axes[2].grid(True,alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("训练曲线已保存为training_curves.png")
    
if __name__=="__main__":
    agent_a,agent_b,rewards,successes,steps=train(episodes=500)
    plot_training_results(rewards, successes, steps)
    print("\n最终统计：")
    print(f"最后100轮成功率：{np.mean(successes[-100:])*100:.1f}%")
    print(f"最佳单轮奖励：{np.max(rewards):.2f}")