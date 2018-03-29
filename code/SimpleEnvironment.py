import numpy as np
import pylab as pl
import itertools
from DiscreteEnvironment import DiscreteEnvironment
from time import sleep
import math
from random import *
import numpy

class SimpleEnvironment(object):
    
    def __init__(self, herb, resolution):
        self.robot = herb.robot
        self.boundary_limits = [[-5., -5.], [5., 5.]]
        self.lower_limits = [-5., -5.]
        self.upper_limits = [5., 5.]
        self.discrete_env = DiscreteEnvironment(resolution, self.lower_limits, self.upper_limits)

        # add an obstacle

        self.table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        self.robot.GetEnv().Add(self.table)

        table_pose = np.array([[ 0, 0, -1, 1.5], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        self.table.SetTransform(table_pose)

        offset = np.zeros(len(self.discrete_env.num_cells))
        offset[0] = 1

        self.offsets = set(itertools.permutations(offset))
        self.draw_counter = 0

        # goal sampling probability
        self.p = 0.0

    def checkCollision(self, coord):

        pose = self.discrete_env.GridCoordToConfiguration(coord)
        # print(pose)

        T = np.array([ [ 1, 0,  0, pose[0,0]], 
                          [ 0, 1,  0, pose[0,1]], 
                          [ 0, 0,  1, 0], 
                          [ 0, 0,  0, 1]])
        self.robot.SetTransform(T)
        return self.robot.GetEnv().CheckCollision(self.table)

    def GetSuccessors(self, grid_coord):
        """
        Returns neighbors of grid_coord. 
        """
        # return np.array([grid_coord + off for off in itertools.permutations(offset)]).astype(np.uint)
        return np.concatenate(([grid_coord + off for off in self.offsets],
            [grid_coord - off for off in self.offsets]), axis = 0).astype(np.uint)

    def ComputeDistance(self, start_coord, end_coord,):
        return np.linalg.norm(self.discrete_env.GridCoordToConfiguration(start_coord) - self.discrete_env.GridCoordToConfiguration(end_coord))    

    def ComputeHeuristicCost(self, start_coord, end_coord):
        #Use distance as heuristic?
        return np.linalg.norm(self.discrete_env.GridCoordToConfiguration(start_coord) - self.discrete_env.GridCoordToConfiguration(end_coord))    



    def InitializePlot(self, goal_config):
        self.fig = pl.figure()
        pl.xlim([self.lower_limits[0], self.upper_limits[0]])
        pl.ylim([self.lower_limits[1], self.upper_limits[1]])
        pl.plot(goal_config[0], goal_config[1], 'gx')

        # Show all obstacles in environment
        for b in self.robot.GetEnv().GetBodies():
            if b.GetName() == self.robot.GetName():
                continue
            bb = b.ComputeAABB()
            pl.plot([bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0]],
                    [bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1]], 'r')
                    
                     
        pl.ion()
        pl.show()


    def PlotEdge(self, sconfig, econfig):
        pl.plot([sconfig[0], econfig[0]],
                [sconfig[1], econfig[1]],
                'k.-', linewidth=2.5)

        if self.draw_counter % 100 == 0:
          pl.draw()
        self.draw_counter = (self.draw_counter + 1)

    def getStatusTable(self):
        return np.full(self.discrete_env.num_cells, False)

    def setDOF(self, values):
        T = np.array([ [ 1, 0,  0, values[0]], 
                          [ 0, 1,  0, values[1]], 
                          [ 0, 0,  1, 0], 
                          [ 0, 0,  0, 1]])
        self.robot.SetTransform(T)
        sleep(0.1)


##############################################################


    def SetGoalParameters(self, goal_config, p = 0.1):
        self.goal_config = goal_config
        self.p = p
        

    def checkCollision_RRT(self, pose):

        T = numpy.array([ [ 1, 0,  0, pose[0]], 
                          [ 0, 1,  0, pose[1]], 
                          [ 0, 0,  1, 0], 
                          [ 0, 0,  0, 1]])
        self.robot.SetTransform(T)
        return self.robot.GetEnv().CheckCollision(self.robot, self.table)


    def GenerateRandomConfiguration(self):
        # Generates and returns a random configuration

        config = [0] * 2;
        lower_limits, upper_limits = self.boundary_limits

        collides = True
        while (collides):
            config[0] = uniform(lower_limits[0], upper_limits[0])
            config[1] = uniform(lower_limits[1], upper_limits[1])
            collides = self.checkCollision_RRT(config)
        
        return numpy.array(config)

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
        # A function which computes the distance between
        # two configurations

        dist = math.sqrt( (start_config[0] - end_config[0])**2 
            + (start_config[1] - end_config[1])**2  )

        return dist

    def clearLineOfSight(self, start_config, end_config, numSamples = 100):
        xvals = numpy.linspace(start_config[0], end_config[0], numSamples)
        yvals = numpy.linspace(start_config[1], end_config[1], numSamples)
        for i in range(0, len(xvals)):
            if self.checkCollision_RRT( [ xvals[i], yvals[i] ] ):
                return False
        return True


    def Extend(self, start_config, end_config, max_extend):
        # A function which attempts to extend from 
        # a start configuration to a goal configuration

        dist = self.ComputeDistance_RRT(start_config, end_config)

        if dist > max_extend:
            u = max_extend / dist     
            new_config = (1-u) * start_config + u * end_config
        else:
            new_config = end_config

        if not self.clearLineOfSight(start_config, new_config):
            return None

        return new_config

    def ShortenPath(self, path, timeout=1.0):
        #  Implemented a function which performs path shortening
        #  on the given path.  Terminate the shortening after the 
        #  given timout (in seconds).

        path = numpy.array(path)

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
                    path = numpy.vstack((path[0:a], pointA, pointB, path[b:]))
                elif a > b:
                    path = numpy.vstack((path[0:b], pointB, pointA, path[a:]))
                else: 
                    print "there's a problem here!"
            
        # self.clearPlot() # UNCOMMENT TO VISUALIZE
        # self.plotPath(path)

        return path

    def getPathLength(self, path):
        
        dist = 0
        for i in range(0,len(path)-1):
            dist += self.ComputeDistance_RRT(path[i], path[i+1])
        return dist


    def PlotPoint(self, config):
        pl.plot(config[0], config[1],
                'g.', markersize=20, )
        pl.draw()


    def PlotEdge_RRT(self, sconfig, econfig):

        pl.plot([sconfig[0], econfig[0]],
                [sconfig[1], econfig[1]],
                'k.-', linewidth=2.5)
        pl.draw()

    def plotPath(self, path, style):

        for i in range(0,len(path)-1):
            pl.plot([path[i][0], path[i+1][0]],
                    [path[i][1], path[i+1][1]],
                    style, linewidth=2.5)
        pl.draw()

    def clearPlot(self):
        pl.cla()
        
