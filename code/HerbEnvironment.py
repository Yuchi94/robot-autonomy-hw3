import numpy as np
from DiscreteEnvironment import DiscreteEnvironment
import itertools

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

    def checkCollision(self, coord):
        config = self.discrete_env.GridCoordToConfiguration(coord)

        self.robot.SetActiveDOFValues(config.squeeze().tolist())
        return self.robot.GetEnv().CheckCollision(self.robot, self.table)

    def GetSuccessors(self, grid_coord):
        """
        Returns neighbors of grid_coord. 
        """
        offset = np.zeros(grid_coord.shape)
        offset[0] = 1

        # return np.array([grid_coord + off for off in itertools.permutations(offset)]).astype(np.uint)
        return np.concatenate(([grid_coord + off for off in itertools.permutations(offset)],
            [grid_coord - off for off in itertools.permutations(offset)]), axis = 0).astype(np.uint)

    def ComputeDistance(self, start_id, end_id):

        return numpy.linalg.norm(self.discrete_env.NodeIdToConfiguration(start_id)
            -self.discrete_env.NodeIdToConfiguration(end_id))

    def ComputeHeuristicCost(self, start_id, goal_id):
        #Use distance as heuristic?
        return numpy.linalg.norm(self.discrete_env.NodeIdToConfiguration(start_id)
            -self.discrete_env.NodeIdToConfiguration(end_id))   

    def getStatusTable(self):
        return np.full(self.discrete_env.num_cells, False)