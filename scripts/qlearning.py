#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../scripts/")
from gridmap import *
from mdp import *
import math
import copy
import random
from matplotlib.animation import PillowWriter    #アニメーション保存用
#%matplotlib notebook


# In[2]:


class QLearning(MDP):
    def __init__(
        self, grid_map_world, drawPI=False, drawQ=False, drawPath=False, drawQPath=False,
        epsilon=1.0, gamma=1.0, alpha=0.5
    ):
        super().__init__(grid_map_world)
        self.cost_map = np.full(self.grid_map.shape, 1)    #その地点が持つコスト
        self.q_map = np.full(self.grid_map.shape+tuple([9]), 0.0)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.drawPIflag = drawPI
        self.drawQflag = drawQ
        self.drawPathflag = drawPath
        self.drawQPathflag = drawQPath
        self.step = 0
        
    def draw(self, ax, elems):
        traversed_grids = self.episode(ax, elems)    #1エピソード実行
        
        #描画
        if(self.drawPIflag):    #方策
            self.drawPI(ax, elems)
        if(self.drawQflag):    #行動価値関数
            self.drawQ(ax, elems)
        if(self.drawQPathflag):    #最大行動価値関数の方策経路
            self.drawQPath(ax, elems)
        if(self.drawPathflag):    #エピソード内の経路
            self.drawTraversedPath(traversed_grids, ax, elems)
    
    #1エピソード実行
    def episode(self, ax, elems):
        s_ = copy.copy(self.world.start_index)
        traversed_grids = []
        for i in range(200):
            s = copy.copy(s_)
            a = self.policy(s_)    #方策
            s_ = self.moveRobot(s, a)   #移動
            self.updateQ(s, a, s_)
            
            if(self.isOutOfBounds(s_) or self.isObstacle(s_) or self.isGoal(s_)):
                break
            traversed_grids.append(copy.copy(s_))
        return traversed_grids
    
    #方策
    def policy(self, s):
        if(random.random() < self.epsilon):
            return self.maxQAgent(s)
        else:
            return self.randomPolicy(s)
        
    #移動
    def moveRobot(self, s, a):
        grid = copy.copy(s)
        if(a == 1 or a == 2 or a == 8):
            grid[0] += 1
        elif(a == 4 or a == 5 or a == 6):
            grid[0] -= 1
        if(a == 2 or a == 3 or a == 4):
            grid[1] += 1
        elif(a == 6 or a == 7 or a == 8):
            grid[1] -= 1
        return grid
   
    #行動価値関数の更新
    def updateQ(self, s, a, s_):
        if(self.isOutOfBounds(s_) or self.isObstacle(s_)):
            q_ = -100.0
        elif(self.isGoal(s_)):
            q_ = 0.0
        else:
            q_ = self.maxQ(s_)
        q = self.Q(s, a)
        self.q_map[s[0]][s[1]][a] = (1- self.alpha) * q + self.alpha * (self.r(s, a, s_) + self.gamma * q_)

    #行動価値関数を最大にする方策を取得
    def maxQAgent(self, s):
        q = self.Q(s, 0)
        pi = 0
        for a in range(9):
            if(q < self.Q(s, a)):
                q_ = self.Q(s, a)
                pi = a
        return pi

    #行動価値関数の最大値
    def maxQ(self, s):
        return np.max(self.q_map[s[0]][s[1]])

    #行動価値関数
    def Q(self, s, a):
        return self.q_map[s[0]][s[1]][a]
  
    def r(self, s, a, s_):
        return self.moveCost(s, s_) - 1.0
        if(self.isGoal(s)):
            return 0.0
        elif(a == 0):
            return -10.0
        else:
            return self.moveCost(s, s_) - 1.0

    def randomPolicy(self, s):
        a = int(random.random() * 8) + 1
        return a
    
    #方策描画
    def drawPI(self, ax, elems):
        for x in range(len(self.grid_map)):
            for y in range(len(self.grid_map[0])):
                if(self.grid_map[x][y] == '0'):
                    continue
                c_num = int(self.maxQAgent([x, y]))
                if(c_num == 0):
                    c = "black"
                elif(c_num == 1):
                    c = "saddlebrown"
                elif(c_num == 2):
                    c = "magenta"
                elif(c_num == 3):
                    c = "blue"
                elif(c_num == 4):
                    c = "cyan"
                elif(c_num == 5):
                    c = "green"
                elif(c_num == 6):
                    c = "lime"
                elif(c_num == 7):
                    c = "yellow"
                elif(c_num == 8):
                    c = "orange"
                elif(c_num == 9):
                    c = "red"
                else:
                    c = "white"
                r = patches.Rectangle(
                    xy=(x*self.world.grid_step[0], y*self.world.grid_step[1]),
                    height=self.world.grid_step[0],
                    width=self.world.grid_step[1],
                    color=c,
                    fill=True,
                    alpha=0.8
                )
                elems.append(ax.add_patch(r))

    #行動価値関数描画
    def drawQ(self, ax, elems):
        for x in range(len(self.grid_map)):
            for y in range(len(self.grid_map[0])):
                if(self.grid_map[x][y] == '0'):
                    continue
                #map1 : 20
                #cost_adj = 7    #map2
                c_num = -self.maxQ([x, y])
                c_num = int(c_num * 7) #Black→Blue
                if(c_num > 0xff): #Blue → Cyan
                    c_num = (c_num-0xff)*16*16 + 0xff
                    if(c_num > 0xffff): #Cyan → Green
                        c_num = 0xffff - int((c_num-0x100ff)*4/256)
                        if(c_num < 0xff00): #Green →Yellow
                            c_num = (0xff00-c_num)*65536+0xff00
                            if(c_num > 0xffff00): #Yellow → Red
                                c_num = 0xffff00 - int((c_num-0xffff00)*0.5/65536)*256
                c = '#' + format(int(c_num), 'x').zfill(6)
                r = patches.Rectangle(
                    xy=(x*self.world.grid_step[0], y*self.world.grid_step[1]),
                    height=self.world.grid_step[0],
                    width=self.world.grid_step[1],
                    color=c,
                    fill=True,
                    alpha=0.8
                )
                elems.append(ax.add_patch(r))
     
    #エピソード内の経路を描画
    def drawTraversedPath(self, grids, ax, elems):
        for grid in grids:
            r = patches.Rectangle(
                xy=(grid[0]*self.world.grid_step[0], grid[1]*self.world.grid_step[1]),
                height=self.world.grid_step[0],
                width=self.world.grid_step[1],
                color="red",
                fill=True,
                alpha=0.5
            )
            elems.append(ax.add_patch(r))
    
    #行動価値関数を最大にする方策で動いた場合の経路を描画
    def drawQPath(self, ax, elems):
        robot_state = copy.copy(self.world.start_index)
        for i in range(100):
            robot_state_old = copy.copy(robot_state)
            a = self.maxQAgent(robot_state)  
            if(a == 1 or a == 2 or a == 8):
                robot_state[0] += 1
            elif(a == 4 or a == 5 or a == 6):
                robot_state[0] -= 1
            if(a == 2 or a == 3 or a == 4):
                robot_state[1] += 1
            elif(a == 6 or a == 7 or a == 8):
                robot_state[1] -= 1
            if(a == 0):
                r = patches.Rectangle(
                    xy=(robot_state[0]*self.world.grid_step[0], robot_state[1]*self.world.grid_step[1]),
                    height=self.world.grid_step[0],
                    width=self.world.grid_step[1],
                    color="black",
                    fill=True,
                    alpha=0.8
                )
                elems.append(ax.add_patch(r))
                break
                
            if(robot_state == self.world.goal_index):
                break
            
            if(self.isOutOfBounds(robot_state) and self.isObstacle(robot_state)):
                break
            
            r = patches.Rectangle(
                xy=(robot_state[0]*self.world.grid_step[0], robot_state[1]*self.world.grid_step[1]),
                height=self.world.grid_step[0],
                width=self.world.grid_step[1],
                color="blue",
                fill=True,
                alpha=0.5
            )
            elems.append(ax.add_patch(r))


# In[4]:


if __name__ == "__main__":
    time_span = 1000
    time_interval = 0.1
    
    grid_step = np.array([0.1, 0.1])
    grid_num = np.array([30, 30])
    
    map_data = "../csvmap/map1.csv"
    
    world = GridMapWorld(grid_step, grid_num, time_span, time_interval, map_data, debug=False)
    world.append(QLearning(world, drawPI=False, drawQ=True, drawPath=False, drawQPath=False, epsilon=0.8))
    
    world.draw()
    #world.ani.save('qlearning_q_map1.gif', writer='pillow', fps=60)    #アニメーション保存


# In[ ]:




