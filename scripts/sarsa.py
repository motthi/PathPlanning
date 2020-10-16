#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../scripts/")
from gridmap import *
from qlearning import *
import math
import copy
import random
from matplotlib.animation import PillowWriter    #アニメーション保存用
#%matplotlib notebook


# In[2]:


class Sarsa(QLearning):
    def __init__(
        self, grid_map_world, drawPI=False, drawQ=False, drawPath=False, drawQPath=False,
        epsilon=1.0, gamma=1.0, alpha=0.5
    ):
        super().__init__(
            grid_map_world, drawPI=False, drawQ=drawQ, drawPath=drawPath, drawQPath=drawQPath,
            epsilon=epsilon, gamma=gamma, alpha=alpha
        )
        
    def draw(self, ax, elems):
        traversed_grids = self.episode(ax, elems)
        if(self.drawPIflag):
            self.drawPI(ax, elems)
        if(self.drawQflag):
            self.drawQ(ax, elems)
        if(self.drawQPathflag):
            self.drawQPath(ax, elems)
        if(self.drawPathflag):
            self.drawTraversedPath(traversed_grids, ax, elems)
            
    #1エピソードを実行
    def episode(self, ax, elems):
        traversed_grids = []
        s, s_ = None, copy.copy(self.world.start_index)
        a, a_ = None, None
        for i in range(200):
            #s_, a_ : 新しい状態での位置と行動
            #s, a : １ステップ前の位置と行動
            if(self.isOutOfBounds(s_) or self.isObstacle(s_) or (self.isGoal(s_))):
                self.updateQ(s, a, s_, a_)
                break
            else:
                a_ = self.policy(s_)    #方策
                self.updateQ(s, a, s_, a_)
                s = copy.copy(s_)
                a = copy.copy(a_)
                s_ = self.moveRobot(s_, a_)   #移動
                traversed_grids.append(copy.copy(s_))
        return traversed_grids
        
    #行動価値関数更新
    def updateQ(self, s, a, s_, a_):
        if s == None: return
        if a == None: return
        if(self.isOutOfBounds(s_) or self.isObstacle(s_)):
            q_ = -100.0
        elif(self.isGoal(s_)):
            q_ = 0.0
        else:
            q_ = self.Q(s_, a_)            
        r = self.r(s, a, s_)
        q = self.Q(s, a)
        self.q_map[s[0]][s[1]][a] = (1- self.alpha) * q + self.alpha * (r + self.gamma * q_)


# In[ ]:


if __name__ == "__main__":
    time_span = 1000
    time_interval = 0.1
    
    grid_step = np.array([0.1, 0.1])
    grid_num = np.array([30, 30])
    
    map_data = "../csvmap/map1.csv"
    
    world = GridMapWorld(grid_step, grid_num, time_span, time_interval, map_data, debug=False)
    world.append(Sarsa(world, drawPI=False, drawQ=False, drawPath=True, drawQPath=True, epsilon=0.9))
    
    world.draw()
    world.ani.save('sarsa_pi_map1.gif', writer='pillow', fps=60)    #アニメーション保存


# In[ ]:




