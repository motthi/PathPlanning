#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("./scripts/")
import math
import numpy as np
from gridmap import*
from robot import*
from matplotlib.animation import PillowWriter    #アニメーション保存用


# In[6]:


class DWA(Robot):
    def __init__(
        self, world, color="black", r=0.02, grid_size_ratio=1,
        noise_per_meter=5, noise_std=np.pi/60, bias_rate_stds=(0.0, 0.0),
        drawPathCandidates_flag=True, drawSelectedPath_flag=True
    ):
        super().__init__(
            None, agent=None, sensor=None, robot_color=color, 
            noise_per_meter=noise_per_meter, noise_std=noise_std, bias_rate_stds=bias_rate_stds,
            expected_stuck_time=1e100, expected_escape_time = 1e-100,
            expected_kidnap_time=1e1000, kidnap_range_x = (0.0,0.0), kidnap_range_y = (0.0,0.0)
        )
        self.world = world
        self.grid_size_ratio = grid_size_ratio
        self.grid_cost_num = self.world.grid_num / self.grid_size_ratio
        self.grid_step = self.world.grid_step * self.grid_size_ratio
        self.start_index = self.indexWorldToCost(self.world.start_index)
        self.goal_index = self.indexWorldToCost(self.world.goal_index)
        
        self.pose = np.append(self.world.start_index * self.world.grid_step + self.world.grid_step / 2, 0.0)
        self.poses = [self.pose]
        
        self.r = r
        self.nu_acc = 0.1
        self.nu_min = 0.
        self.nu_max = 0.5
        self.nu_delta = 0.1
        self.omega_acc = 50*np.pi/180
        self.omega_min = -100*np.pi / 180
        self.omega_max = 100*np.pi / 180
        self.omega_delta = 5*np.pi/180
        
        self.time_interval = self.world.time_interval
        self.nu = 0.0
        self.omega = 0.0
        
        self.drawPathCandidates_flag = drawPathCandidates_flag
        self.drawSelectedPath_flag = drawSelectedPath_flag
    
    def initialize(self):
        self.nu = 0.0
        self.omega = 0.0
        self.pose = np.append(self.world.start_index * self.world.grid_step + self.world.grid_step / 2, 0.0)
        self.poses = self.pose
    
    def draw(self, ax, elems):
        self.drawRobot(self.pose, ax, elems)
        self.drawTakenPath(self.poses, ax, elems)
        if self.isRobotInCostGoal(self.pose) or self.isRobotInWorldObstacle(self.pose):
            return
        
        control_inputs = self.controlInputCandidate()
        path_candidates = self.pathCandidate(control_inputs)
        nu, omega = self.selectPath(path_candidates)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)
        self.nu, self.omega = nu, omega
        
        self.drawPathCandidates(path_candidates, ax, elems) if self.drawPathCandidates_flag is True else None
        self.drawSelectedPath(self.pose, self.nu, self.omega, ax, elems) if self.drawSelectedPath_flag is True else None
        
        self.pose = self.state_transition(self.nu, self.omega, self.world.time_interval, self.pose)
        self.poses.append(self.pose)
    
    def run(self):
        self.initialize()
        while not self.isRobotInCostGoal(self.pose) and not self.isRobotInWorldObstacle:
            self.pose = self.next(self.pose)
            self.poses.append(self.pose)
    
    def next(self, pose):
        control_inputs = self.controlInputCandidate()
        path_candidates = self.pathCandidate(control_inputs)
        self.nu, self.omega = self.selectPath(path_candidates)
        self.nu, self.omega = self.considerUncertainty(self.nu, self.omega, self.world.time_interval)
        next_pose = self.state_transition(self.nu, self.omega, self.world.time_interval, self.pose)
        return next_pose
    
    def plot(self, figsize=(4, 4), save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.world.grid_step[0] * self.world.grid_num[0])
        ax.set_ylim(0, self.world.grid_step[1] * self.world.grid_num[1])
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        # Map
        for index, grid in np.ndenumerate(self.world.grid_map):
            if grid == '0':  #Obstacle
                self.world.drawGrid(index, "black", 1.0, ax)
            if grid == '2' or self.world.isStart(index):  #Start
                self.world.drawGrid(index, "orange", 1.0, ax)
            elif grid == '3' or self.world.isGoal(index):  #Goal 
                self.world.drawGrid(index, "green", 1.0, ax)
                
        #Path
        ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        plt.show()

        if(save_path is not None):
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        return fig
    
    def controlInputCandidate(self):
        controlInput = []
        for omega in np.arange(self.omega-self.omega_acc, self.omega+self.omega_acc, self.omega_delta):
            for nu in np.arange(self.nu-self.nu_acc, self.nu+self.nu_acc+1e-5, self.nu_delta):
                if (nu < self.nu_max and nu > self.nu_min) and (omega < self.omega_max and omega > self.omega_min):
                    controlInput.append([nu, omega])
        return controlInput
    
    def pathCandidate(self, control_inputs):
        candidate_path = []
        for control_input in control_inputs:
            xs, ys, thetas = self.pose
            nu, omega = control_input
            path = [[xs, ys, thetas, 0.0, 0.0]]
            flag=0
            for i in range(5):
                xf, yf, thetaf = self.state_transition(nu, omega, self.world.time_interval, np.array([xs, ys, thetas]))
                if self.isRobotInWorldObstacle([xf, yf, thetaf]):
                    flag=1
                    break
                path.append([xf, yf, thetaf, nu, omega])
                xs, ys, thetas = xf, yf, thetaf
            candidate_path.append(path) if flag==0 else None
        return candidate_path
    
    def selectPath(self, path_candidates):
        cost = float('inf')
        select_nu, select_omega = 0.0, 0.0
        for path in path_candidates:
            x, y, theta, nu, omega = path[-1]
            c = 0.1*self.costObstacle([x, y, theta]) + 20 * self.costSpeed(nu) + 50*self.costGoalDistance([x, y, theta])
            if(c < cost):
                cost = c
                select_nu, select_omega = nu, omega
        return select_nu, select_omega
    
    def costObstacle(self, pose):
        index = self.poseToWorldIndex(pose)
        theta = pose[2]
        cost = 0.0
        for i in range(-2, 4):
            for j in range(-2, 4):
                if not self.isInRange(np.arctan2(j, i), theta-45, theta+45):
                    continue
                sensing_index = index + np.array([i, j])
                if self.world.isOutOfBounds(sensing_index):
                    continue
                if self.world.isObstacle(sensing_index):
                    cost += 1.0 / (np.linalg.norm(pose[0:2] - self.worldIndexToPose(sensing_index)[0:2])+1e-5)
        return cost
    
    def costSpeed(self, nu):
        return 1.0 / (nu+1e-5)
    
    def costGoalDistance(self, pose):
        cost = np.linalg.norm(pose[0:2] - self.indexToPose(self.world.goal_index)[0:2])
        return cost
    
    def indexWorldToCost(self, index):
        return np.array(index) // self.grid_size_ratio
    
    def indexCostToWorld(self, index):
        return np.array(index) * self.grid_size_ratio
         
    def poseToWorldIndex(self, pose):
        return (np.array(pose[0:2]) // self.world.grid_step).astype(np.int32)
    
    def poseToCostIndex(self, pose):
        return (np.array(pose[0:2]) // self.grid_step).astype(np.int32)
    
    def costIndexToPose(self, index):
        return np.append(index * self.grid_step + self.grid_step / 2, 0.0)
    
    def worldIndexToPose(self, index):
        return np.append(index * self.world.grid_step + self.world.grid_step / 2, 0.0)
    
    def isRobotInCostGoal(self, pose):
        index = self.poseToCostIndex(pose)
        if np.all(index == self.goal_index):
            return True
        else:
            return False
    
    def isRobotInWorldObstacle(self, pose):
        index = self.poseToCostIndex(pose)
        index = self.indexCostToWorld(index)
        if self.world.isObstacle(index):
            return True
        for grid in neigbor_grids:
            neigbor_index = index + grid
            if self.world.isObstacle(neigbor_index):
                neigbor_pose = self.worldIndexToPose(neigbor_index)
                dis_pose = np.abs(neigbor_pose - pose)[0:2] - self.world.grid_step / 2
                if np.all(grid == [1, 0]) or np.all(grid == [-1, 0]):
                    dis = dis_pose[0]
                elif np.all(grid == [0, 1]) or np.all(grid == [0, -1]):
                    dis = dis_pose[1]
                elif np.all(grid == [1, 1]) or np.all(grid == [-1, -1]) or np.all(grid == [1, -1]) or np.all(grid == [-1, 1]):
                    if dis_pose[0] < self.r and dis_pose[1] < self.r:
                        return True
                    else:
                        continue
                if dis < self.r:
                    return True
        return False
    
    def isRobotInCostObstacle(self, pose):
        index = self.poseToCostIndex(pose)
        if self.hasObstacle(index):
            return True
        for grid in neigbor_grids:
            neigbor_index = index + grid
            if self.hasObstacle(neigbor_index):
                neigbor_pose = self.costIndexToPose(neigbor_index)
                dis_pose = np.abs(neigbor_pose - pose)[0:2] - self.grid_step / 2
                if np.all(grid == [1, 0]) or np.all(grid == [-1, 0]):
                    dis = dis_pose[0]
                elif np.all(grid == [0, 1]) or np.all(grid == [0, -1]):
                    dis = dis_pose[1]
                elif np.all(grid == [1, 1]) or np.all(grid == [-1, -1]) or np.all(grid == [1, -1]) or np.all(grid == [-1, 1]):
                    if dis_pose[0] < self.r and dis_pose[1] < self.r:
                        return True
                    else:
                        continue
                if dis < self.r:
                    return True
        return False
    
    def isCostGridOutOfBounds(self, index):
        if np.any(index >= self.grid_cost_num) or np.any(index < [0, 0]):
            return True
        else:
            return False
    
    def drawPathCandidates(self, path_candidates, ax, elems=None):
        for path in path_candidates:
            for i in range(len(path) - 1):
                xs, ys, _, _, _ = path[i]
                xf, yf, _, _, _ = path[i+1]
                if elems is None:
                    ax.plot([xs, xf], [ys, yf], color="cyan", lineWidth=0.4)
                else:
                    elems += ax.plot([xs, xf], [ys, yf], color="cyan", lineWidth=0.4)
    
    def drawSelectedPath(self, pose, nu, omega, ax, elems):
        xs, ys, thetas = pose
        for i in range(5):
            xf, yf, thetaf = self.state_transition(nu, omega, self.world.time_interval, np.array([xs, ys, thetas]))
            if elems is None:
                ax.plot([xs, xf], [ys, yf], color="red", lineWidth=1.0)
            else:
                elems += ax.plot([xs, xf], [ys, yf], color="red", lineWidth=1.0)
            xs, ys, thetas = xf, yf, thetaf


# In[7]:


if __name__ == "__main__":
    time_span = 15
    time_interval = 0.1

    grid_step = np.array([0.1, 0.1])
    grid_num = np.array([30, 30])

    map_data = "./csvmap/map4.csv"

    world = GridMapWorld(grid_step, grid_num, time_span, time_interval, map_data, time_show="time", debug=False)
    #world.append(DWA(world, color="red", r=0.05, noise_per_meter=10, noise_std=np.pi/30))
    world.append(DWA(world, r=0.08, grid_size_ratio=2, noise_per_meter=0, noise_std=0.0))
    
    world.draw()
    #world.ani.save('dwa_map7.gif', writer='ffmpeg', fps=100)    #アニメーション保存


# In[ ]:




