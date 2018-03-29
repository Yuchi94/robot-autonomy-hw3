import numpy
import matplotlib.pyplot as pl
from random import *
import math
import time

import pdb

class RRT_SimpleEnvironment(object):
    
    def __init__(self, herb):
        self.robot = herb.robot
        self.boundary_limits = [[-5., -5.], [5., 5.]]

        # add an obstacle
        global table
        table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        self.robot.GetEnv().Add(table)

        table_pose = numpy.array([[ 0, 0, -1, 1.0], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        table.SetTransform(table_pose)

        # goal sampling probability
        self.p = 0.0

    def SetGoalParameters(self, goal_config, p = 0.1):
        self.goal_config = goal_config
        self.p = p
        

    def checkCollision(self, pose):
        # T = numpy.eye(4)
        # T[0,3] = pose[0] #x
        # T[0,3] = pose[1] #y

        T = numpy.array([ [ 1, 0,  0, pose[0]], 
                          [ 0, 1,  0, pose[1]], 
                          [ 0, 0,  1, 0], 
                          [ 0, 0,  0, 1]])
        self.robot.SetTransform(T)
        return self.robot.GetEnv().CheckCollision(self.robot, table)


    def GenerateRandomConfiguration(self):
        # Generates and returns a random configuration

        config = [0] * 2;
        lower_limits, upper_limits = self.boundary_limits

        collides = True
        while (collides):
            config[0] = uniform(lower_limits[0], upper_limits[0])
            config[1] = uniform(lower_limits[1], upper_limits[1])
            collides = self.checkCollision(config)
        
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

    def ComputeDistance(self, start_config, end_config):
        # A function which computes the distance between
        # two configurations

        dist = math.sqrt( (start_config[0] - end_config[0])**2 
            + (start_config[1] - end_config[1])**2  )

        return dist

    def clearLineOfSight(self, start_config, end_config, numSamples = 100):
        xvals = numpy.linspace(start_config[0], end_config[0], numSamples)
        yvals = numpy.linspace(start_config[1], end_config[1], numSamples)
        for i in range(0, len(xvals)):
            if self.checkCollision( [ xvals[i], yvals[i] ] ):
                return False
        return True


    def Extend(self, start_config, end_config, max_extend):
        # A function which attempts to extend from 
        # a start configuration to a goal configuration

        dist = self.ComputeDistance(start_config, end_config)

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
            dist += self.ComputeDistance(path[i], path[i+1])
        return dist


    def InitializePlot(self, goal_config):
        self.fig = pl.figure()
        lower_limits, upper_limits = self.boundary_limits
        pl.xlim([lower_limits[0], upper_limits[0]])
        pl.ylim([lower_limits[1], upper_limits[1]])
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
        
    def PlotPoint(self, config):
        pl.plot(config[0], config[1],
                'g.', markersize=20, )
        pl.draw()

    def PlotEdge(self, sconfig, econfig):

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