# -*- coding: utf-8 -*-
"""
Q-learning智能体
"""

import numpy as np
import random

class QLearningAgent:
    #初始化智能体
    def __init__(self,env,agent_id,learning_rate=0.1,discount=0.9,epsilon=0.2):
        """
        env:环境对象
        agent_id:'A'或'B'
        learning_rate:学习率（0.1=每次学10%的新经验）
        discount:折扣因子（0.9=重视未来90%的奖励）
        epsilon:探索率（0.2=20%的时间随机乱走）
        """
        self.env=env
        self.id=agent_id
        #设置起点和终点
        if agent_id=='A':
            self.start=env.start_A
            self.goal=env.goal_A
        else:
            self.start=env.start_B
            self.goal=env.goal_B
        #当前位置
        self.x,self.y=self.start
        #动作空间：0=上，1=下，2=左，3=右，4=停留
        self.actions=[0,1,2,3,4]
        self.action_names=['上','下','左','右','停留']
        self.num_actions=5
        #超参数
        self.lr=learning_rate
        self.gamma=discount
        self.epsilon=epsilon
        #Q表（经验备忘录）：（10，10，5），初始全为0，即什么都不知道
        self.Q_table=np.zeros((env.size,env.size,self.num_actions))
        #统计信息
        self.steps_taken=0
        self.collisions=0 
        
    #获取当前位置
    def get_state(self):
        return (self.x,self.y)
    
    #选择动作
    def choose_action(self,training=True):
        """
        training=True:探索模式（部分随机）
        training=False:利用模式（完全按经验）
        """
        if self.has_reached_goal():
            return 4
        state=self.get_state()
        x,y=state
        if training and random.random()<self.epsilon:
            #探索：随机选动作
            action=random.choice(self.actions)
        else:
            #利用：选Q值最大的动作
            action=np.argmax(self.Q_table[x,y,:])
        return action
    
    #根据动作计算下一步的位置
    def get_next_position(self,action):
        x,y=self.x,self.y
        if action==0:#上
            x-=1 
        elif action==1:#下
            x+=1 
        elif action==2:#左
            y-=1 
        elif action==3:#右
            y+=1 
        elif action==4:#停留
            pass
        return (x,y)
    
    #计算奖励
    def calculate_reward(self,new_pos,reached_goal,collision):
        if reached_goal:
            return 10.0 
        elif collision:
            return -5.0 
        else:
            return -0.1 
        
    #更新Q表
    def update_Q(self,state,action,reward,next_state,done):
        """
        更新公式：
        Q(s,a) = Q(s,a) + α × [r + γ × maxQ(s') - Q(s,a)]
        """
        x,y=state
        nx,ny=next_state
        #当前Q值
        current_q=self.Q_table[x,y,action]
        if done:
            #游戏结束，没有未来奖励
            target=reward
        else:
            #未来最大Q值
            next_max_q=np.max(self.Q_table[nx,ny,:])
            target=reward+self.gamma*next_max_q
        #更新Q值
        new_q=current_q+self.lr*(target-current_q)
        self.Q_table[x,y,action]=new_q
        
    #重置到起点
    def reset(self):
        self.x,self.y=self.start
        self.steps_taken=0 
        self.collisions=0 
        
    #设置位置
    def set_position(self,x,y):
        self.x,self.y=x,y
        self.steps_taken+=1 
        
    #检查是否到达终点
    def has_reached_goal(self):
        return (self.x,self.y)==self.goal
    
    #打印当前位置的Q值（测试）
    def print_Q_at_position(self):
        x,y=self.x,self.y
        print(f"位置（{x},{y}）的Q值：")
        for i,name in enumerate(self.action_names):
            print(f"{name}：{self.Q_table[x,y,i]:.2f}")
            
#测试
if __name__=="__main__":
    from environment import Environment
    env=Environment()
    agent=QLearningAgent(env, 'A')
    print(f"智能体{agent.id}创建成功")
    print(f"起点：{agent.start}，终点：{agent.goal}")
    print(f"Q表形状：{agent.Q_table.shape}")
    print(f"动作空间：{agent.action_names}")
    print("\n测试动作选择：")
    print(f"当前状态：{agent.get_state()}")
    print(f"选择动作：{agent.action_names[agent.choose_action()]}")