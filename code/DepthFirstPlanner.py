import numpy as np
from collections import deque

class DepthFirstPlanner(object):
    
    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        self.nodes = dict()

    def Plan(self, start_config, goal_config):
                
        # TODO: Here you will implement the depth first planner
        #  The return path should be a numpy array
        #  of dimension k x n where k is the number of waypoints
        #  and n is the dimension of the robots configuration space
        # print(start_config)

        status = self.planning_env.getStatusTable()
        start_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(start_config)

        goal_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(goal_config)
        neighbors = self.planning_env.GetSuccessors(start_coord)

        queue = deque()
        parents = {}
        parents[tuple(start_coord)] = None
        status[np.split(start_coord, start_coord.shape[0])] = True


        for n in neighbors:
            status[np.split(n, n.shape[0])] = True
            queue.append(n)
            parents[tuple(n)] = start_coord

        while True:
            #choose node from right side of deque
            node = queue.pop()

            #Add neighbors
            neighbors = self.planning_env.GetSuccessors(node)
            for n in neighbors:
                #OOB
                if (n < 0).any() or (n >= self.planning_env.discrete_env.num_cells).any():
                    continue

                #Explored
                if status[np.split(n, n.shape[0])] == True:
                    continue

                #Collision
                if self.planning_env.checkCollision(n):
                    continue

                #Reached the end
                if (n == goal_coord).all():
                    parents[tuple(n)] = node
                    return self.createPath(start_config, goal_config, parents, n)
                    
                #Mark square as visited
                status[np.split(n, n.shape[0])] = True
                queue.append(n)

                parents[tuple(n)] = node
        
        print("Should never reach here")

    def createPath(self, start, goal, parents, coord):
        path = []
        path.append(np.expand_dims(goal, axis = 0))

        while True:
            path.append(self.planning_env.discrete_env.GridCoordToConfiguration(coord))
            coord = parents[tuple(coord)]
            if coord is None:
                break

        path.append(np.expand_dims(start, axis = 0))
        return np.concatenate(path[::-1], axis = 0)