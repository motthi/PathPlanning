#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import warnings


# In[2]:


class GridBasePathPlanning():
    def __init__(self, world, grid_size_ratio=1):
        self.world = world
        self.grid_size_ratio = grid_size_ratio
        self.grid_cost_num = self.world.grid_num / self.grid_size_ratio
        if not self.grid_cost_num[0].is_integer() or not self.grid_cost_num[1].is_integer():
            warnings.warn("World's grid map and DstarLite's grid cost map are incompatible")
        self.grid_cost_num = self.grid_cost_num.astype('int8')
    
    def drawCostSizeGrid(self, index, color, alpha, ax, fill=True, elems=None):
        xy = index * self.world.grid_step * self.grid_size_ratio
        r = patches.Rectangle(
            xy=(xy),
            height=self.world.grid_step[0] * self.grid_size_ratio,
            width=self.world.grid_step[1] * self.grid_size_ratio,
            color=color,
            alpha=alpha,
            fill=fill
        )
        if elems is not None:
            elems.append(ax.add_patch(r))
        else:
            ax.add_patch(r)
            
    def indexWorldToCost(self, index):
        return np.array(index) // self.grid_size_ratio
    
    def indexCostToWorld(self, index):
        return np.array(index) * self.grid_size_ratio
    
    def isCostGridOutOfBounds(self, index):
        if np.any(index >= self.grid_cost_num) or np.any(index < [0, 0]):
            return True
        else:
            return False
    
    def hasStart(self, index):
        index = self.indexCostToWorld(index)
        for i in range(self.grid_size_ratio):
            for j in range(self.grid_size_ratio):
                if self.world.isStart([index[0]+i, index[1]+j]):
                    return True
        return False
    
    def hasGoal(self, index):
        index = self.indexCostToWorld(index)
        for i in range(self.grid_size_ratio):
            for j in range(self.grid_size_ratio):
                if self.world.isGoal([index[0]+i, index[1]+j]):
                    return True
        return False
    
    def hasObstacle(self, index):
        if np.any(index >= self.grid_cost_num) or np.any(index < [0, 0]):
            return True
        index = self.indexCostToWorld(index)
        for i in range(self.grid_size_ratio):
            for j in range(self.grid_size_ratio):
                if self.world.isObstacle([index[0]+i, index[1]+j]):
                    return True
        return False


# In[3]:


class PathNotCalculatedError(Exception):
    pass

class PathNotFoundError(Exception):
    pass

