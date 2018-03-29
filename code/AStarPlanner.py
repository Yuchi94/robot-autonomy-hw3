import numpy as np
from collections import deque
from itertools import count
from heapq import *
np.set_printoptions(threshold=np.inf)

class AStarPlanner(object):
    
    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        self.nodes = dict()


    def Plan(self, start_config, goal_config):

        start_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(start_config)
        goal_coord = self.planning_env.discrete_env.ConfigurationToGridCoord(goal_config)
        neighbors = self.planning_env.GetSuccessors(start_coord)

        if self.visualize:
            self.planning_env.InitializePlot(goal_config)

        h = [] #Use a heap
        parents = {}
        parents[tuple(start_coord)] = (None, 0)
        curr_cost = {}
        tb = count()

        for n in neighbors:
            heappush(h, (1 + self.planning_env.ComputeHeuristicCost(n, goal_coord), 
                        1, next(tb), n)) #Total cost, current cost, tiebreaker num, coord
            parents[tuple(n)] = (start_coord, 1)

        while True:
            #pop min element from heap
            node = heappop(h)

            #Add neighbors
            neighbors = self.planning_env.GetSuccessors(node[3])

            for n in neighbors:
                #OOB
                if (n < 0).any() or (n >= self.planning_env.discrete_env.num_cells).any():
                    continue

                #Explored
                if tuple(n) in parents:
                    continue
                    if parents[tuple(n)][1] <= node[1] + self.planning_env.ComputeDistance(node[3], n):
                        continue
                    else:
                        for H in h:
                            if (H[3] == n).all():
                                h.remove(H)
                                break
                        heapify(h)


                #Collision
                if self.planning_env.checkCollision(n):
                    continue

                #visualize
                if self.visualize:
                    self.planning_env.PlotEdge(np.squeeze(self.planning_env.discrete_env.GridCoordToConfiguration(node[3])).copy(), 
                        np.squeeze(self.planning_env.discrete_env.GridCoordToConfiguration(n)).copy())

                #Reached the end
                if (n == goal_coord).all():
                    parents[tuple(n)] = (node[3], node[1] + self.planning_env.ComputeDistance(node[3], n))
                    return self.createPath(start_config, goal_config, parents, n)
                
                #Add parents
                heappush(h, (node[1] + self.planning_env.ComputeDistance(node[3], n) 
                    + self.planning_env.ComputeHeuristicCost(n, goal_coord), 
                    node[1] + self.planning_env.ComputeDistance(node[3], n), next(tb), n))
                parents[tuple(n)] = (node[3], node[1] + self.planning_env.ComputeDistance(node[3], n))
        
        print("Should never reach here")

    def createPath(self, start, goal, parents, coord):
        print("number of explored nodes: " + str(len(parents)))

        path = []
        path.append(np.expand_dims(goal, axis = 0))

        while True:
            path.append(self.planning_env.discrete_env.GridCoordToConfiguration(coord))
            coord = parents[tuple(coord)][0]
            if coord is None:
                break

        path.append(np.expand_dims(start, axis = 0))
        return np.concatenate(path[::-1], axis = 0)
