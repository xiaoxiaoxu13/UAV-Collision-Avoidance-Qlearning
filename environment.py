# -*- coding: utf-8 -*-
"""
定义环境：地图、障碍物、起点、终点
"""

class Environment:
    #初始化地图
    def __init__(self):
        self.size=10
        
        #定义障碍物位置
        self.obstacles=[
            (2,2),(2,3),(3,2),(3,3),
            (5,7),(6,7),(7,7),
            (8,4)]
        
        #起点和终点
        self.start_A=(1,1)
        self.goal_A=(8,8)
        self.start_B=(8,9)
        self.goal_B=(1,1)
        
    #检查pos处是否是障碍物
    def is_obstacle(self,pos):
        return pos in self.obstacles
    
    #检查pos处是否在地图内
    def is_valid(self,pos):
        x,y=pos
        return 0<=x<self.size and 0<=y<self.size
    
    #检查两个位置是否相同（碰撞）
    def is_collision(self,pos1,pos2):
        return pos1==pos2
    
    #检查位置是否是该智能体的终点
    def is_goal(self,pos,agent_id):
        if agent_id=='A':
            return pos==self.goal_A
        else:
            return pos==self.goal_B
        
    #打印地图（测试）
    def print_map(self):
        print("\n地图布局（A=起点A，a=终点A，B=起点B，b=终点B，*=障碍物，.=空地）：")
        print("-"*30)
        for i in range(self.size):
            row=""
            for j in range(self.size):
                if (i,j) in self.obstacles:
                    row+="* "
                elif (i,j) ==self.start_A:
                    row+="A "
                elif (i,j) ==self.goal_A:
                    row+="a "
                elif (i,j) ==self.start_B:
                    row+="B "
                elif (i,j) ==self.goal_B:
                    row+="b "
                else:
                    row+="."
            print(row)
        print("-"*30)

#测试        
if __name__=="__main__":
    env=Environment()
    print("环境创建成功！")
    print(f"地图大小：{env.size} x {env.size}")
    print(f"障碍物数量：{len(env.obstacles)}")
    print(f"A起点：{env.start_A}->A终点：{env.goal_A}")
    print(f"B起点：{env.start_B}->B终点：{env.goal_B}")
    env.print_map()