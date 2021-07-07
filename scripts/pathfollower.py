#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("./scripts/")
import numpy as np
from gridmap import*
from robot import*
from astar import*
from dstar import*
from dstarlite import*
from bug import*
from sensor import*
from dijkstra import PathNotCalculatedError
from math import exp
from matplotlib.animation import PillowWriter    #アニメーション保存用


# In[2]:


class PathFollower(Robot):
    def __init__(
        self, world, pp_algorithm, robot_color="black", path_color="red", r=0.02,
        noise_per_meter=5, noise_std=np.pi/60, bias_rate_stds=(0.0, 0.0),
        drawWayPoint_flag=False, drawMetricMap_flag=False
    ):
        super().__init__(
            None, agent=None, sensor=None, robot_color=robot_color, path_color=path_color, r=r,
            noise_per_meter=noise_per_meter, noise_std=noise_std, bias_rate_stds=bias_rate_stds,
            expected_stuck_time=1e100, expected_escape_time = 1e-100,
            expected_kidnap_time=1e1000, kidnap_range_x = (0.0,0.0), kidnap_range_y = (0.0,0.0)
        )
        self.world = world
        self.time = 0
        self.time_interval = self.world.time_interval
        self.pp_algorithm = pp_algorithm
        self.grid_size_ratio = self.pp_algorithm.grid_size_ratio
        self.grid_cost_num = self.pp_algorithm.grid_cost_num
        self.grid_step = self.world.grid_step * self.grid_size_ratio
        
        #if np.any(self.world.start_index == None) or np.any(self.world.goal_index == None):
        #    theta = 0.0
        #else:
        self.start_index = self.indexWorldToCost(self.world.start_index)
        self.goal_index = self.indexWorldToCost(self.world.goal_index)
        d_pose = self.goal_index - self.start_index
        theta = np.arctan2(d_pose[1], d_pose[0])
        
        self.followIndex = self.world.start_index
        
        self.pose = np.append(self.start_index * self.grid_step+ self.grid_step / 2, theta)
        self.poses = [self.pose]
        
        self.nu = 0.0
        self.nu_min = 0.0
        self.nu_max = 0.4
        self.nu_delta = 0.01
        
        self.omega = 0.0
        self.omega_min = -320*np.pi / 180
        self.omega_max = 320*np.pi / 180
        self.omega_delta = 10*np.pi / 180
        
        self.way_point = self.goal_index
        self.flag_chng_index = True
        
        self.drawWayPoint_flag = drawWayPoint_flag
        self.drawMetricMap_flag = drawMetricMap_flag
    
    def draw(self, ax, elems):
        self.drawRobot(self.pose, ax, elems)
        self.drawTakenPath(self.poses, ax, elems, linewidth=1.0, label=self.pp_algorithm.pp_algorithm_name)
        
        if self.isRobotInGoal(self.pose):
            pass
        elif self.isRobotInWorldObstacle(self.pose):
            self.drawCostSizeGrid(self.poseToCostIndex(self.pose), "red", 0.5, ax, elems=elems)
        else:
            self.pose = self.next(self.pose)
            self.poses.append(self.pose)
            self.drawCostSizeGrid(self.way_point, "blue", 0.5, ax, elems=elems)  if self.drawWayPoint_flag else None
            if hasattr(self.pp_algorithm, "checkNewObstacle"):
                self.pp_algorithm.drawNewObstacles(ax, elems) if not self.pp_algorithm.drawMetricMap_flag else None
            ax.legend(fontsize=15, framealpha=1.0, loc="upper right")
    
    def run(self):
        cnt = 0
        d_pose = self.world.goal_index - self.world.start_index
        theta = np.arctan2(d_pose[1], d_pose[0])
        self.pose = np.append(self.world.start_index * self.world.grid_step + self.world.grid_step / 2, theta)
        chk_counter = 10
        while not self.isRobotInCostGoal(self.pose) and not self.isRobotInWorldObstacle(self.pose):
            self.pose = self.next(self.pose)
            self.poses.append(self.pose)
            cnt += 1
            
            next_index = self.poseToCostIndex(self.pose)
            if not np.all(current_index == next_index):
                current_index = next_index
                idx_history.append(list(current_index))
                if idx_history.count(list(current_index)) > chk_counter:
                    break
        self.time = self.time_interval * cnt
    
    def plot(self, figsize=(4, 4), color="red", save_path=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.world.grid_step[0] * self.world.grid_num[0])
        ax.set_ylim(0, self.world.grid_step[1] * self.world.grid_num[1])
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        # Map
        for index, grid in np.ndenumerate(self.world.grid_map):
            if grid == '0':
                self.world.drawGrid(index, "black", 1.0, ax)
            if grid == '2' or self.world.isStart(index):  #Start
                self.world.drawGrid(index, "orange", 1.0, ax)
            elif grid == '3' or self.world.isGoal(index):  #Goal 
                self.world.drawGrid(index, "green", 1.0, ax)
        
        self.drawTakenPath(self.poses, ax, linewidth=1.0, label="BUG")
        
        plt.show()

        if(save_path is not None):
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        return fig
    
    def next(self, pose):
        index = self.poseToCostIndex(pose)
        next_index = index
        
        if hasattr(self.pp_algorithm, "checkNewObstacle"):
            self.pp_algorithm.checkNewObstacle(next_index)
        
        if self.flag_chng_index == True:
            prev_next = next_index
            while np.linalg.norm(self.pose[0:2] - self.costIndexToPose(next_index)[0:2]) < np.linalg.norm(self.grid_step)*0.5:
                try:
                    next_index = self.pp_algorithm.next(next_index)
                except PathNotCalculatedError:
                    self.pp_algorithm.run()
                    next_index = self.pp_algorithm.next(self.start_index)
                if np.any(next_index == None):
                    next_index = self.followIndex
                if self.hasGoal(next_index):
                    break
                if np.all(index == next_index):
                    next_index = prev_next
                    break
                prev_next = next_index
            self.followIndex = next_index
        else:
            next_index = self.way_point
            
        if np.all(np.abs(next_index - index) == [1, 1]):
            if np.all(next_index - index == [1, 1]):
                if self.hasObstacle(index + [1, 0]):
                    next_index = index + [0, 1]
                elif self.hasObstacle(index + [0, 1]):
                    next_index = index + [1, 0]
            elif np.all(next_index - index == [-1, 1]):
                if self.hasObstacle(index + [-1, 0]):
                    next_index = index + [0, 1]
                elif self.hasObstacle(index + [0, 1]):
                    next_index = index + [-1, 0]
            elif np.all(next_index - index == [1, -1]):
                if self.hasObstacle(index + [1, 0]):
                    next_index = index + [0, -1]
                elif self.hasObstacle(index + [0, -1]):
                    next_index = index + [1, 0]
            elif np.all(next_index - index == [-1, -1]):
                if self.hasObstacle(index + [-1, 0]):
                    next_index = index + [0, -1]
                elif self.hasObstacle(index + [0, -1]):
                    next_index = index + [-1, 0]
        
        if self.flag_chng_index == True:
            self.way_point = next_index
        
        #control_inputs = self.controlInputCandidate()
        #path_candidates = self.pathCandidate(control_inputs)
        d_theta = np.arctan2(self.costIndexToPose(next_index)[1] - pose[1], self.costIndexToPose(next_index)[0] - pose[0]) - pose[2]
        while d_theta > np.pi:
            d_theta -= 2*np.pi
        while d_theta < -np.pi:
            d_theta += 2*np.pi
        if d_theta < -np.pi/6:
            nu, omega = 0.0, -320*np.pi/180
        elif d_theta > np.pi/6:
            nu, omega = 0.0, 320*np.pi/180
        else:
            nu, omega = self.nu_max * exp(-d_theta), d_theta * 5
            #nu, omega = self.selectPath(path_candidates, goalH=0.01)
        next_pose = self.state_transition(nu, omega, self.world.time_interval, pose)
        
        if np.linalg.norm(next_pose[0:2] - self.costIndexToPose(next_index)[0:2]) > np.linalg.norm(self.grid_step)/4:
            self.flag_chng_index = False
        else:
            self.flag_chng_index = True
        return  next_pose
    
    def controlInputCandidate(self):
        controlInput = []
        for omega in np.arange(self.omega_min, self.omega_max, self.omega_delta):
            controlInput.append([self.nu_max, omega])
            controlInput.append([self.nu_max/2, omega])
            controlInput.append([0, omega])
            #controlInput.append([0.0, omega])
            #for nu in np.arange(self.nu-self.nu_acc, self.nu+self.nu_acc+1e-5, self.nu_delta):
            #    if (nu < self.nu_max and nu > self.nu_min) and (omega < self.omega_max and omega > self.omega_min):
            #        controlInput.append([nu, omega])
        return controlInput
    
    def pathCandidate(self, control_inputs):
        candidate_path = []
        for control_input in control_inputs:
            xs, ys, thetas = self.pose
            nu, omega = control_input
            path = [[xs, ys, thetas, 0.0, 0.0]]
            flag=0
            for i in range(3):
                xf, yf, thetaf = self.state_transition(nu, omega, self.world.time_interval, np.array([xs, ys, thetas]))
                if self.isRobotInObstacle([xf, yf, thetaf]):
                    flag=1
                    break
                path.append([xf, yf, thetaf, nu, omega])
                xs, ys, thetas = xf, yf, thetaf
            candidate_path.append(path) if flag==0 else None
        return candidate_path
    
    def selectPath(self, path_candidates, speed=10.0, goalH=0.0, goalD=1.0, obs=0.0):
        cost = float('inf')
        c_goalH_c = float('inf')
        c_goalD_c = float('inf')
        select_nu, select_omega = 0.0, 0.0
        for path in path_candidates:
            x, y, theta, nu, omega = path[-1]
            c_speed = self.costSpeed(nu)
            c_goalH = self.costGoalHeading([x, y, theta])
            c_goalD = self.costGoalDistance([x, y, theta])
            c_obs = self.costObstacle([x, y, theta]) if obs != 0.0 else 0.0
            c = c_speed*c_speed + goalH*c_goalH + goalD*c_goalD
            if(c < cost):
                cost = c
                select_nu, select_omega = nu, omega
        return select_nu, select_omega
    
    def costObstacle(self, pose):
        index = self.poseToIndex(pose)
        theta = pose[2]
        cost = 0.0
        for i in range(-1, 3):
            for j in range(-1, 3):
                if not self.isInRange(np.arctan2(j, i), theta-45, theta+45):
                    continue
                sensing_index = index + np.array([i, j])
                if self.world.isOutOfBounds(sensing_index):
                    cost += 1.0
                if self.world.isObstacle(sensing_index):
                    cost += 1.0 / (np.linalg.norm(pose[0:2] - self.indexToPose(sensing_index)[0:2])+1e-5)
        return cost
    
    def costSpeed(self, nu):
        return self.nu_max - nu
    
    def costGoalDistance(self, pose):
        cost = np.linalg.norm(pose[0:2] - self.indexToPose(self.way_point)[0:2])
        return cost
    
    def costGoalHeading(self, pose):
        theta = np.arctan2(self.indexToPose(self.way_point)[0:2][1] - pose[0:2][1], self.indexToPose(self.way_point)[0:2][0] - pose[0:2][0])
        d_theta = abs(pose[2] - theta)
        return d_theta
    
    def indexWorldToCost(self, index):
        return np.array(index) // self.grid_size_ratio
    
    def indexCostToWorld(self, index):
        return np.array(index) * self.grid_size_ratio
    
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


# In[3]:


class BugFollower(PathFollower):
    def __init__(
        self, world, grid_size_ratio=1, r=0.02,
        noise_per_meter=5, noise_std=np.pi/60, bias_rate_stds=(0.0, 0.0),
        robot_color="black", path_color="red",
        drawWayPoint_flag=False, drawMLine_flag=False
    ):
        super().__init__(
            world, BUG(world, grid_size_ratio), robot_color=robot_color, path_color=path_color, r=r,
            noise_per_meter=5, noise_std=np.pi/60, bias_rate_stds=(0.0, 0.0),
            drawWayPoint_flag=drawWayPoint_flag
        )
        self.m_line = np.array([])
        self.drawMLine_flag = drawMLine_flag
    
    def initialize(self):
        self.m_line = np.array([])
        self.alongObstacle = False
        self.prev_index = None
        self.prev_next_index = None
        self.prev_obs = None
        self.v = np.array([])
        self.takenPath = []
        
        d_pose = self.goal_index - self.start_index
        theta = np.arctan2(d_pose[1], d_pose[0])
        self.pose = np.append(self.start_index * self.grid_step + self.grid_step / 2, theta)
        self.poses = [self.pose]
        self.getMLine()
        self.mLine_min = float('inf')
        self.taken_mLine = [self.start_index]
        self.flag_chng_index = False
        self.way_point = self.goal_index
    
    def plot(self, figsize=(4, 4), color="red", save_path=None, drawMLine_flag=False):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.world.grid_step[0] * self.world.grid_num[0])
        ax.set_ylim(0, self.world.grid_step[1] * self.world.grid_num[1])
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        # Map
        for index, grid in np.ndenumerate(self.world.grid_map):
            if grid == '0':
                self.world.drawGrid(index, "black", 1.0, ax)
            if grid == '2' or self.world.isStart(index):  #Start
                self.world.drawGrid(index, "orange", 1.0, ax)
            elif grid == '3' or self.world.isGoal(index):  #Goal
                self.world.drawGrid(index, "green", 1.0, ax)
        
        if drawMLine_flag is True:
            for index in self.m_line:
                if not self.hasStart(index) and not self.hasGoal(index):
                    self.drawCostSizeGrid(index, "lime", 0.5, ax)
        
        self.drawTakenPath(self.poses, ax, linewidth=1.0, label="BUG")
        
        plt.show()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        return fig
    
    def run(self):
        cnt = 0
        chk_counter = 10
        idx_history = [list(self.poseToCostIndex(self.pose))]
        if len(self.m_line) == 0:
            self.initialize()
        while not self.isRobotInWorldObstacle(self.pose) and not self.isRobotInCostGoal(self.pose):
            self.pose = self.next(self.pose)
            self.poses.append(self.pose)
            cnt += 1
            
            next_index = self.poseToCostIndex(self.pose)
            if not np.all(current_index == next_index):
                current_index = next_index
                idx_history.append(list(current_index))
                if idx_history.count(list(current_index)) > chk_counter:
                    break
        self.time = self.time_interval * cnt
    
    def draw(self, ax, elems):
        self.drawRobot(self.pose, ax, elems)
        self.drawTakenPath(self.poses, ax, elems, linewidth=1.0, label="BUG")
        
        if len(self.m_line) == 0:
            self.initialize()
            self.drawMLine(ax) if(self.drawMLine_flag is True) else None
            
        if self.isRobotInCostGoal(self.pose):
            pass
        elif self.isRobotInWorldObstacle(self.pose):
            self.drawCostSizeGrid(self.poseToCostIndex(self.pose), "red", 0.5, ax, elems=elems)
        else:
            self.pose = self.next(self.pose)
            self.poses.append(self.pose)
            self.drawCostSizeGrid(self.way_point, "blue", 0.5, ax, elems=elems) if self.drawWayPoint_flag else None
        ax.legend(fontsize=15, framealpha=1.0, loc="upper right")
    
    def next(self, pose):
        index = self.poseToCostIndex(pose)
        next_pose = None
        
        if len(self.m_line) == 0:
            raise PathNotCalculatedError("Path did not calculate")
        
        # 過去に通過したM-Lineよりも近いM-Lineにに到達した場合
        if np.any(np.all(self.m_line == index, axis=1)) and not np.any(np.all(self.taken_mLine == index, axis=1)) and np.linalg.norm(self.costIndexToPose(self.goal_index)[0:2] - pose[0:2]) <= self.mLine_min+1e-3:
            self.alongObstacle = False
            self.taken_mLine.append(index)
        
        # 前回，障害物がなかった場合
        if self.alongObstacle is False:
            dx, dy = index - self.goal_index
            theta_g = np.arctan2(dy, dx)
            theta_sort = np.abs((theta_g - theta_neigbors + 2 * np.pi) % (2 * np.pi) - np.pi)
            neigbor_grid = np.argsort(theta_sort)[0]
            next_index = index + neigbor_grids[neigbor_grid]
            
            d_theta = np.arctan2(self.goal_index[1] - index[1], self.goal_index[0] - index[0]) - pose[2]
            while d_theta > np.pi:
                d_theta -= 2*np.pi
            while d_theta < -np.pi:
                d_theta += 2*np.pi
            if d_theta < -np.pi/6:
                nu, omega = 0.1, -320*np.pi / 180
            elif d_theta > np.pi/6:
                nu, omega = 0.1, 320*np.pi / 180
            else:
                nu, omega = self.nu_max * exp(-d_theta), d_theta * 5
            next_pose = self.state_transition(nu, omega, self.time_interval, pose)
            
            # ゴールに直線的に移動中に障害物に衝突した場合，way pointを変更
            if self.isRobotInCostObstacle(next_pose):
                chk_index = self.poseToCostIndex(next_pose)
                for grid in neigbor_grids:
                    neigbor_index = chk_index + grid
                    if self.hasObstacle(neigbor_index):
                        neigbor_pose = self.costIndexToPose(neigbor_index)
                        dis_pose = np.abs(neigbor_pose - next_pose)[0:2] - self.grid_step / 2
                        if np.all(grid == [1, 0]) or np.all(grid == [-1, 0]):
                            dis = dis_pose[0]
                        elif np.all(grid == [0, 1]) or np.all(grid == [0, -1]):
                            dis = dis_pose[1]
                        elif np.all(grid == [1, 1]) or np.all(grid == [-1, -1]) or np.all(grid == [1, -1]) or np.all(grid == [-1, 1]):
                            if dis_pose[0] < self.r and dis_pose[1] < self.r:
                                next_index = neigbor_index
                                break
                            else:
                                continue
                        if dis < self.r:
                            next_index = neigbor_index
                            break
                
                self.alongObstacle = True
                self.flag_chng_index = True
                next_index = self.moveCW(index, next_index)
                self.way_point = next_index
                d_theta = np.arctan2(next_index[1] - index[1], next_index[0] - index[0]) - pose[2]
                while d_theta > np.pi:
                    d_theta -= 2*np.pi
                while d_theta < -np.pi:
                    d_theta += 2*np.pi
                if d_theta < -np.pi/6:
                    nu, omega = 0.0, -320*np.pi/180
                elif d_theta > np.pi/6:
                    nu, omega = 0.0, 320*np.pi/180
                else:
                    nu, omega = self.nu_max * exp(-d_theta), d_theta * 5
                    #control_inputs = self.controlInputCandidate()
                    #path_candidates = self.pathCandidate(control_inputs)
                    #nu, omega = self.selectPath(path_candidates, speed=10.0, goalH=0.00, goalD=1.0)
                next_pose = self.state_transition(nu, omega, self.time_interval, pose)
            else:
                self.mLine_min = np.linalg.norm(self.costIndexToPose(self.goal_index)[0:2] - pose[0:2])
            self.v = index - next_index
            return next_pose
        
        if self.flag_chng_index:
            if self.hasObstacle(self.prev_obs - self.v):
                next_index = self.moveCW(index, self.prev_obs - self.v)
            else:
                next_index = self.prev_obs - self.v
                if np.all(np.abs(next_index - index) == [1, 1]):
                    next_index = index - self.v
                if self.hasObstacle(next_index):
                    next_index = self.moveCW(index, next_index)
            self.v = index - next_index
            self.way_point = next_index
        else:
            next_index = self.way_point
        
        d_theta = np.arctan2(self.costIndexToPose(next_index)[1] - pose[1], self.costIndexToPose(next_index)[0] - pose[0]) - pose[2]
        while d_theta > np.pi:
            d_theta -= 2*np.pi
        while d_theta < -np.pi:
            d_theta += 2*np.pi
        if d_theta < -np.pi/6:
            nu, omega = 0.0, -320*np.pi/180
        elif d_theta > np.pi/6:
            nu, omega = 0.0, 320*np.pi/180
        else:
            nu, omega = self.nu_max * exp(-d_theta), d_theta * 5
            #control_inputs = self.controlInputCandidate()
            #path_candidates = self.pathCandidate(control_inputs)
            #nu, omega = self.selectPath(path_candidates, speed=10.0, goalH=0.0001, goalD=1.0)
        next_pose = self.state_transition(nu, omega, self.time_interval, pose)
        
        if np.linalg.norm(next_pose[0:2] - self.costIndexToPose(next_index)[0:2]) > np.linalg.norm(self.grid_step)/4:
            self.flag_chng_index = False
        else:
            self.flag_chng_index = True
            
        return next_pose
    
    def controlInputCandidate(self):
        controlInput = []
        for omega in np.arange(self.omega_min, self.omega_max, self.omega_delta):
            controlInput.append([self.nu_max, omega])
            #controlInput.append([self.nu_max/2, omega])
            controlInput.append([0, omega])
            #controlInput.append([0.0, omega])
            #for nu in np.arange(self.nu-self.nu_acc, self.nu+self.nu_acc+1e-5, self.nu_delta):
            #    if (nu < self.nu_max and nu > self.nu_min) and (omega < self.omega_max and omega > self.omega_min):
            #        controlInput.append([nu, omega])
        return controlInput
    
    def pathCandidate(self, control_inputs):
        candidate_path = []
        for control_input in control_inputs:
            xs, ys, thetas = self.pose
            nu, omega = control_input
            path = [[xs, ys, thetas, 0.0, 0.0]]
            flag=0
            for i in range(1):
                xf, yf, thetaf = self.state_transition(nu, omega, self.time_interval, np.array([xs, ys, thetas]))
                if self.isRobotInObstacle([xf, yf, thetaf]):
                    flag=1
                    break
                path.append([xf, yf, thetaf, nu, omega])
                xs, ys, thetas = xf, yf, thetaf
            candidate_path.append(path) if flag==0 else None
        return candidate_path
    
    def getMLine(self):
        s = self.start_index
        g = self.goal_index
        theta = np.arctan2(g[1] - s[1], g[0] - s[0])
        self.m_line = np.array([s])
        distance = 0.0
        index = s
        while not np.all(index == self.goal_index):
            distance += np.linalg.norm(self.grid_step) * 0.25
            pos = self.costIndexToPose(s)[0:2] + np.array([np.cos(theta), np.sin(theta)]) * distance
            index = self.poseToCostIndex(pos)
            if not np.any([np.all(index == grid) for grid in self.m_line]):
                self.m_line = np.append(self.m_line, self.poseToCostIndex(pos).reshape(1, 2), axis=0)
    
    def moveCW(self, index, next_chk_index):# 左回りになるように移動
        dxy = index - next_chk_index
        if np.all(dxy == [1, 0]) or np.all(dxy == [1, 1]): # 障害物が左側または左下側の場合，下へ移動する
            if not self.hasObstacle(index + [0, -1]):
                next_index = index + [0, -1]
                self.prev_obs = index + [-1, 0]
            elif not self.hasObstacle(index + [1, 0]):
                next_index = index + [1, 0]
                self.prev_obs = index + [0, -1]
            elif not self.hasObstacle(index + [0, 1]):
                next_index = index + [0, 1]
                self.prev_obs = index + [1, 0]
            else:
                next_index = index + [-1, 0]
        elif np.all(dxy == [1, -1]) or np.all(dxy == [0, -1]): # 障害物が左上側または上側の場合，左へ移動する
            if not self.hasObstacle(index + [-1, 0]):
                next_index = index + [-1, 0]
                self.prev_obs = index + [0, 1]
            elif not self.hasObstacle(index + [0, -1]):
                next_index = index + [0, -1]
                self.prev_obs = index + [-1, 0]
            elif not self.hasObstacle(index + [1, 0]):
                next_index = index + [1, 0]
                self.prev_obs = index + [0, -1]
            else:
                next_index = index + [0, 1]
        elif np.all(dxy == [-1, 0]) or np.all(dxy == [-1, -1]): # 障害物が右側または右上側の場合，上へ移動する
            if not self.hasObstacle(index + [0, 1]):
                next_index = index + [0, 1]
                self.prev_obs = index + [1, 0]
            elif not self.hasObstacle(index + [-1, 0]):
                next_index = index + [-1, 0]
                self.prev_obs = index + [0, 1]
            elif not self.hasObstacle(index + [0, -1]):
                next_index = index + [0, -1]
                self.prev_obs = index + [-1, 0]
            else:
                next_index = index + [1, 0]
        elif np.all(dxy == [-1, 1]) or np.all(dxy == [0, 1]): # 障害物が下側または左下側の場合，右へ移動する
            if not self.hasObstacle(index + [1, 0]):
                next_index = index + [1, 0]
                self.prev_obs = index + [0, -1]
            elif not self.hasObstacle(index + [0, 1]):
                next_index = index + [0, 1]
                self.prev_obs = index + [1, 0]
            elif not self.hasObstacle(index + [-1, 0]):
                next_index = index + [-1, 0]
                self.prev_obs = index + [0, 1]
            else:
                next_index = index + [0, 1]
        return next_index
    
    def drawMLine(self, ax):
        for line in self.m_line:
            if not(self.hasStart(line) or self.hasGoal(line)):
                self.drawCostSizeGrid(line, "lime", 0.5, ax)


# In[5]:


if __name__ == "__main__":
    time_span = 60
    time_interval = 0.1

    grid_step = np.array([0.1, 0.1])
    grid_num = np.array([30, 30])

    map_data = "./csvmap/map2.csv"

    world = GridMapWorld(grid_step, grid_num, time_span, time_interval, map_data, time_show="time", debug=False, is_dynamic=False)
    print(world.start_index, world.goal_index)
    
    pp_algorithm = Astar(world, grid_size_ratio=1)
    world.append(PathFollower(world, pp_algorithm, r=0.03, path_color="red", drawWayPoint_flag=False))
    
    pp_algorithm = DstarLite(world, IdealSensor(world, sensing_range=5), grid_size_ratio=1, drawLocalMap_flag=False)
    pp_algorithm.initialize()
    world.append(PathFollower(world, pp_algorithm, r=0.03, path_color="blue"))
    
    bug_follower = BugFollower(world, grid_size_ratio=1, r=0.03, path_color="green", drawWayPoint_flag=False, drawMLine_flag=False)
    world.append(bug_follower)
    
    world.draw(figsize=(8, 8))
    #world.ani.save('path_45_compare3.gif', writer='ffmpeg', fps=100)    #アニメーション保存


# In[ ]:




