# -*- coding: utf-8 -*-
"""
测试训练好的模型并可视化路径
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from agent import QLearningAgent
from train import execute_actions

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#测试训练好的智能体，不探索
def test(agent_a,agent_b,env,render=True):
    """
    返回：
    trajectory_a:A的轨迹
    trajectory_b:B的轨迹
    success:是否成功
    steps:所用步数
    """
    agent_a.reset()
    agent_b.reset()
    #记录轨迹
    trajectory_a=[(agent_a.x,agent_a.y)]
    trajectory_b=[(agent_b.x,agent_b.y)]
    max_steps=200
    done=False
    steps_taken=max_steps
    for step in range(max_steps):
        #最优策略，不探索
        action_a=agent_a.choose_action(training=False)
        action_b=agent_b.choose_action(training=False)
        #执行动作
        new_a_pos,new_b_pos,_,_,done=execute_actions(agent_a, agent_b, env, action_a, action_b)
        agent_a.set_position(*new_a_pos)
        agent_b.set_position(*new_b_pos)
        trajectory_a.append((agent_a.x,agent_a.y))
        trajectory_b.append((agent_b.x,agent_b.y))
        if done:
            steps_taken=step+1
            if render:
                print(f"测试成功！用了{step+1}步")
            break
    success=done
    if not success and render:
        print("测试失败：未能在步数内到达目标")
    return trajectory_a,trajectory_b,success,steps_taken

#可视化路径
def visualize_path(trajectory_a,trajectory_b,env):
    plt.figure(figsize=(10,10))
    plt.xlim(-0.5,env.size-0.5)
    plt.ylim(env.size-0.5,-0.5)
    #灰色网格线
    for i in range(env.size+1):
        plt.axhline(i-0.5,color='gray',linewidth=0.5)
        plt.axvline(i-0.5,color='gray',linewidth=0.5)
    #黑色障碍物
    for obs in env.obstacles:
        row,col=obs
        plt.fill_between([col-0.5,col+0.5], row-0.5, row+0.5,color='black',alpha=0.8)
    #A的起点：绿色方块，A的终点：红色方块
    ay,ax=env.start_A
    plt.fill_between([ax-0.4,ax+0.4],ay-0.4,ay+0.4,color='lightgreen',edgecolor='green',linewidth=2)
    ay,ax=env.goal_A
    plt.fill_between([ax-0.4,ax+0.4],ay-0.4,ay+0.4,color='salmon',edgecolor='red',linewidth=2)
    #B的起点：浅蓝色方块，B的终点：黄色方块
    by, bx = env.start_B
    plt.fill_between([bx - 0.4, bx + 0.4], by - 0.4, by + 0.4, color='lightblue', edgecolor='blue', linewidth=2)
    by, bx = env.goal_B
    plt.fill_between([bx - 0.4, bx + 0.4], by - 0.4, by + 0.4, color='gold', edgecolor='orange', linewidth=2)
    #提取轨迹坐标
    traj_a_y=[p[0] for p in trajectory_a]
    traj_a_x=[p[1] for p in trajectory_a]
    traj_b_y=[p[0] for p in trajectory_b]
    traj_b_x=[p[1] for p in trajectory_b]
    #绘制路径
    plt.plot(traj_a_x,traj_a_y,'r-',linewidth=2.5,alpha=0.8,label='无人机A路径',marker='o',markersize=4,markevery=10)
    plt.plot(traj_b_x,traj_b_y,'b-',linewidth=2.5,alpha=0.8,label='无人机B路径',marker='s',markersize=4,markevery=10)
    #标记起点和终点
    ay, ax = env.start_A
    plt.text(ax, ay, 'A起点', fontsize=10, ha='center', va='bottom')
    ay, ax = env.goal_A
    plt.text(ax, ay, 'A终点', fontsize=10, ha='center', va='top')
    by, bx = env.start_B
    plt.text(bx, by, 'B起点', fontsize=10, ha='center', va='top')
    by, bx = env.goal_B
    plt.text(bx, by, 'B终点', fontsize=10, ha='center', va='bottom')
    #添加图例
    plt.legend(loc='upper right', fontsize=12)
    #添加标题
    plt.title('双无人机路径规划结果\n（红色=无人机A，蓝色=无人机B）', fontsize=14)
    #添加网格标签
    plt.xlabel('列 (x坐标)', fontsize=12)
    plt.ylabel('行 (y坐标)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('path_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("路径图已保存为 path_visualization.png")
    
#分析最终策略
def analyze_strategy(agent_a,agent_b,env):
    print("\n"+"="*50)
    print("最终策略分析")
    print("="*50)
    print("\n无人机A在起点(1,1)的Q值:")
    agent_a.x, agent_a.y = 1, 1
    agent_a.print_Q_at_position()
    print("\n无人机B在起点(8,9)的Q值:")
    agent_b.x, agent_b.y = 8, 9
    agent_b.print_Q_at_position()
    
#多次测试评估成功率
def run_multiple_tests(agent_a,agent_b,env,num_tests=100):
    successes=0
    steps_list=[]
    for i in range(num_tests):
        _,_,success,steps=test(agent_a,agent_b,env,render=False)
        if success:
            successes+=1
            steps_list.append(steps)
    success_rate=successes/num_tests*100
    avg_steps=np.mean(steps_list) if steps_list else float('inf')
    print(f"\n===测试评估（{num_tests}次）===")
    print(f"成功率：{success_rate:.1f}%")
    print(f"平均完成步数：{avg_steps:.1f}")
    return success_rate,avg_steps

if __name__=="__main__":
    env=Environment()
    env.print_map()
    agent_a=QLearningAgent(env, 'A')
    agent_b=QLearningAgent(env, 'B')
    try:
        agent_a.Q_table=np.load('q_table_a.npy')
        agent_b.Q_table=np.load('q_table_b.npy')
        print("\n成功加载训练好的Q表")
    except FileNotFoundError:
        print("\n错误：未找到Q表文件，请先运行train.py训练模型")
        exit(1)
    #单次测试并可视化
    print("\n执行单次测试……")
    traj_a,traj_b,success,steps=test(agent_a,agent_b,env)
    visualize_path(traj_a, traj_b, env)
    #多次测试评估
    run_multiple_tests(agent_a, agent_b, env)
    #策略分析
    analyze_strategy(agent_a, agent_b, env)
    print("\n"+"="*50)
    print("测试完成！生成的文件：")
    print("-path_visualization.png:路径可视化图")
    print("-training_curves.png:训练曲线（已存在）")
    print("="*50)