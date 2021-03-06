import numpy as np
from collections import deque

class DepthFirstPlanner(object):
    
    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        self.nodes = dict()

    def Plan(self, start_config, goal_config):
        start_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(start_config)
        goal_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(goal_config)
        neighbors = self.planning_env.GetSuccessors(start_coord)

        queue = deque()
        parents = {}
        parents[tuple(start_coord)] = None

        if self.visualize:
            self.planning_env.InitializePlot(goal_config)

        for n in neighbors:
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
                if tuple(n) in parents:
                    continue

                #Collision
                if self.planning_env.checkCollision(n):
                    continue
                #visualize
                if self.visualize:
                    self.planning_env.PlotEdge(np.squeeze(self.planning_env.discrete_env.GridCoordToConfiguration(node)).copy(), 
                        np.squeeze(self.planning_env.discrete_env.GridCoordToConfiguration(n)).copy())

                #Reached the end
                if (n == goal_coord).all():
                    parents[tuple(n)] = node
                    return self.createPath(start_config, goal_config, parents, n)
                    
                #Add parents
                queue.append(n)
                parents[tuple(n)] = node
        
        print("Should never reach here")

    def createPath(self, start, goal, parents, coord):
        print("number of explored nodes: " + str(len(parents)))

        path = []
        path.append(np.expand_dims(goal, axis = 0))

        while True:
            path.append(self.planning_env.discrete_env.GridCoordToConfiguration(coord))
            coord = parents[tuple(coord)]
            if coord is None:
                break

        path.append(np.expand_dims(start, axis = 0))
        return np.concatenate(path[::-1], axis = 0)