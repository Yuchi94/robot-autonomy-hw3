import numpy as np
from DiscreteEnvironment import DiscreteEnvironment
import itertools
# from time import sleep
import sys
import matplotlib.pyplot as pl
from random import *
import math
import time

class HerbEnvironment(object):
    
    def __init__(self, herb, resolution):
        
        self.robot = herb.robot
        self.lower_limits, self.upper_limits = self.robot.GetActiveDOFLimits()
        self.discrete_env = DiscreteEnvironment(resolution, self.lower_limits, self.upper_limits)

        # account for the fact that snapping to the middle of the grid cell may put us over our
        #  upper limit
        upper_coord = [x - 1 for x in self.discrete_env.num_cells]
        upper_config = self.discrete_env.GridCoordToConfiguration(upper_coord)
        for idx in range(len(upper_config)):
            self.discrete_env.num_cells[idx] -= 1

        # add a table and move the robot into place
        self.table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        
        self.robot.GetEnv().Add(self.table)

        table_pose = np.array([[ 0, 0, -1, 0.7], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        self.table.SetTransform(table_pose)
        
        # set the camera
        camera_pose = np.array([[ 0.3259757 ,  0.31990565, -0.88960678,  2.84039211],
                                   [ 0.94516159, -0.0901412 ,  0.31391738, -0.87847549],
                                   [ 0.02023372, -0.9431516 , -0.33174637,  1.61502194],
                                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.robot.GetEnv().GetViewer().SetCamera(camera_pose)

        offset = np.zeros(len(self.discrete_env.num_cells))
        offset[0] = 1
        self.offsets = set(itertools.permutations(offset))

        self.p = 0.0

    def checkCollision(self, coord):
        robot_saver = self.robot.CreateRobotStateSaver(
              self.robot.SaveParameters.ActiveDOF
            | self.robot.SaveParameters.ActiveManipulator
            | self.robot.SaveParameters.LinkTransformation)
        limits = self.robot.GetActiveDOFLimits()
        config = self.discrete_env.GridCoordToConfiguration(coord)
        env = self.robot.GetEnv()

        with robot_saver, env:
            self.robot.SetActiveDOFValues(config.squeeze().tolist())
            return env.CheckCollision(self.robot) or (limits[0] > config).any() or (limits[1] < config).any()

    def GetSuccessors(self, grid_coord):
        """
        Returns neighbors of grid_coord. 
        """
        return np.concatenate(([grid_coord + off for off in self.offsets],
            [grid_coord - off for off in self.offsets]), axis = 0).astype(np.uint)

    def ComputeDistance(self, start_coord, end_coord):

        return np.linalg.norm(self.discrete_env.GridCoordToConfiguration(start_coord) - self.discrete_env.GridCoordToConfiguration(end_coord))    

    def ComputeHeuristicCost(self, start_coord, end_coord):
        #Use distance as heuristic?
        return np.linalg.norm(self.discrete_env.GridCoordToConfiguration(start_coord) - self.discrete_env.GridCoordToConfiguration(end_coord))    

    def getStatusTable(self):
        return np.full(self.discrete_env.num_cells, False)

    def setDOF(self, values):
        self.robot.SetActiveDOFValues(values.squeeze().tolist())
        sleep(0.1)

##########################################################################
### RRT SPECFIC FUNCTIONS ################################################ 


    def SetGoalParameters(self, goal_config, p = 0.2):
        self.goal_config = goal_config
        self.p = p
        
    def checkCollision_RRT(self, config):
        self.robot.SetActiveDOFValues(config)
        return self.robot.GetEnv().CheckCollision(self.robot, self.table)

    def GenerateRandomConfiguration(self):
        lower_limits, upper_limits = self.robot.GetActiveDOFLimits()
        while True:
            config = np.random.uniform(lower_limits, upper_limits, len(self.robot.GetActiveDOFIndices()))
            if not self.checkCollision_RRT(config):
                break

        return np.array(config)

    def GenerateRandomConfiguration_GoalBias(self, goal_config = None):
        
        if goal_config is None:
            goal = self.goal_config
        else:
            goal = goal_config

        if uniform(0,1) < self.p:
            return goal
        else:
            return self.GenerateRandomConfiguration()
    
    def ComputeDistance_RRT(self, start_config, end_config):
        
        return np.linalg.norm(np.array(start_config)-np.array(end_config))


    def Extend(self, start_config, end_config, max_extend):

        end_config = (end_config - start_config) / self.ComputeDistance_RRT(end_config, start_config) * max_extend + start_config
        
        numSamples = 100
        vals = np.array([np.linspace(i, j, numSamples) for i, j in zip(start_config, end_config)]).T

        for v in vals:
            if self.checkCollision_RRT(v):
                return None

        return end_config
        
    def ShortenPath(self, path, timeout=1.0):
        #  Implemented a function which performs path shortening
        #  on the given path.  Terminate the shortening after the 
        #  given timout (in seconds).

        path = np.array(path)

        start_time = time.time()
        while(time.time() - start_time < timeout and len(path) != 2):

            # sample point A
            a = randint(1, len(path)-2)
            u = uniform(0,1)   
            pointA = (1-u) * path[a-1] + u * path[a]

            # sample point B
            b = a
            while (b == a): # make sure we don't get the same point
                b = randint(1, len(path)-2)   
            u = uniform(0,1) # where to sample interpolated point
            pointB = (1-u) * path[b-1] + u * path[b]

            # self.clearPlot() # UNCOMMENT TO VISUALIZE
            # self.plotPath(path)
            # self.PlotPoint(pointA)
            # self.PlotPoint(pointB)
            # self.PlotEdge(pointA,pointB)

            # is there a collision free LOS? 
            if self.clearLineOfSight(pointA, pointB):

                if a < b: # trim from a to b
                    path = np.vstack((path[0:a], pointA, pointB, path[b:]))
                elif a > b:
                    path = np.vstack((path[0:b], pointB, pointA, path[a:]))
                else: 
                    print "there's a problem here!"
            
        # self.clearPlot() # UNCOMMENT TO VISUALIZE
        # self.plotPath(path)

        return path

    def clearLineOfSight(self, start_config, end_config, numSamples = 1000):
        vals = np.array([np.linspace(i, j, numSamples) for i, j in zip(start_config, end_config)]).T

        for v in vals:
            if self.checkCollision_RRT(v):
                return False

        return True

    def PlotEdge_RRT(self, sconfig, econfig):
        # placeholder
        pass

    def plotPath(self, arga, argb):
        #dummy function
        pass

    def getPathLength(self, path):
        
        dist = 0
        for i in range(0,len(path)-1):
            dist += self.ComputeDistance_RRT(path[i], path[i+1])
        return dist