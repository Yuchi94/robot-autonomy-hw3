import numpy
from RRTTree import RRTTree
import time

class HeuristicRRTPlanner(object):

    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        

    def Plan(self, start_config, goal_config, epsilon = 0.3):
        
        tree = RRTTree(self.planning_env, start_config)
        plan = []
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)
        # TODO: Here you will implement the rrt planner
        #  The return path should be an array
        #  of dimension k x n where k is the number of waypoints
        #  and n is the dimension of the robots configuration space
        
        heappush(h, (1 + self.planning_env.ComputeHeuristicCost(n, goal_coord), 
            1, next(tb), n)) # Total cost, current cost, tiebreaker num, coord

        heappush(h, (node[1] + self.planning_env.ComputeDistance(node[3], n) 
            + self.planning_env.ComputeHeuristicCost(n, goal_coord), 
            node[1] + self.planning_env.ComputeDistance(node[3], n), next(tb), n))

        #  Implementation of the rrt planner
        #  The return path should be an array
        #  of dimension k x n where k is the number of waypoints
        #  and n is the dimension of the robots configuration space

        start_time = time.time() # Start Timer

        goalDist = self.planning_env.ComputeDistance(start_config, goal_config)
        while goalDist > epsilon:
            
            # SELECT NODE
            new_config = self.planning_env.GenerateRandomConfiguration_GoalBias()
            closest_id, closest_config = tree.GetNearestVertex(new_config)
            
            # EXTEND
            new_config = self.planning_env.Extend(closest_config, new_config, epsilon)

            if new_config is not None:

                new_id = tree.AddVertex(new_config)
                tree.AddEdge(closest_id, new_id)
                self.planning_env.PlotEdge(new_config, closest_config) # PLOT

                closest_goal_id, closest_goal_config = tree.GetNearestVertex(goal_config)
                goalDist = self.planning_env.ComputeDistance(closest_goal_config, goal_config)
                # print goalDist

        goal_id = tree.AddVertex(goal_config)
        tree.AddEdge(closest_goal_id, goal_id)

        ### traverse home! ###

        parent_id = goal_id 
        parent_config = goal_config

        while parent_id is not None: 
            
            plan.append(parent_config)
            child_id = parent_id
            parent_id, parent_config = tree.getParent(child_id)
        
        plan = plan[::-1]

        end_time = time.time() - start_time
        print "ok we're done, time to plan:", end_time

        return plan
