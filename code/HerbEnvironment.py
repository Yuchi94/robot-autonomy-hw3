import numpy as np
from DiscreteEnvironment import DiscreteEnvironment
import itertools
from time import sleep
import sys

class HerbEnvironment(object):
    
    def __init__(self, herb, resolution):
        
        self.robot = herb.robot
        self.lower_limits, self.upper_limits = self.robot.GetActiveDOFLimits()
        print(self.lower_limits)
        print(self.upper_limits)
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

    def checkCollision(self, coord):
        robot_saver = self.robot.CreateRobotStateSaver(
              self.robot.SaveParameters.ActiveDOF
            | self.robot.SaveParameters.ActiveManipulator
            | self.robot.SaveParameters.LinkTransformation)
        limits = self.robot.GetActiveDOFLimits()
        config = self.discrete_env.GridCoordToConfiguration(coord)
        env = self.robot.GetEnv()
        if (self.lower_limits > config).any() or (self.upper_limits < config).any():
            return True

        with robot_saver, env:
            self.robot.SetActiveDOFValues(config.squeeze().tolist())
            return env.CheckCollision(self.robot)

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
