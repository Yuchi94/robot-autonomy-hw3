import numpy as np
import pylab as pl
import itertools
from DiscreteEnvironment import DiscreteEnvironment
from time import sleep

class SimpleEnvironment(object):
    
    def __init__(self, herb, resolution):
        self.robot = herb.robot
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

        
