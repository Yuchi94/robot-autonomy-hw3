import numpy
from RRTTree import RRTTree
import time
from random import *
import pdb

class HeuristicRRTPlanner(object):

    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        

    def Plan(self, start_config, goal_config, epsilon = 0.3):
        
        goalDist = self.planning_env.ComputeDistance_RRT(start_config, goal_config)
        C_opt = goalDist
        
        tree = RRTTree(self.planning_env, start_config, C_opt)
        plan = []
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)

        start_time = time.time() # Start Timer

        while goalDist > epsilon:
            
            # SELECT NODE
            new_config = self.planning_env.GenerateRandomConfiguration_GoalBias()
            closest_id, closest_config, closest_cost = tree.GetNearestVertex(new_config)
                
            HEURISTIC = self.planning_env.ComputeDistance_RRT(new_config, goal_config)
            
            # PATH_COST = self.planning_env.ComputeDistance_RRT(closest_config, new_config)
            
            PATH_COST = 0
            child_config = new_config
            parent_id = closest_id 
            parent_config = closest_config
            while parent_id is not None: 
                PATH_COST = PATH_COST + self.planning_env.ComputeDistance_RRT(parent_config, child_config)
                child_id = parent_id
                child_config = parent_config
                parent_id, parent_config = tree.getParent(child_id)

            C_vertex = HEURISTIC + PATH_COST

            m = 1 - (C_vertex - C_opt) / (tree.get_max_cost() - C_opt)
            p = max(m,0.01)

            # print "Cost",C_vertex,"m",m,"p",p

            if uniform(0,1) < p:
                # EXTEND
                new_config = self.planning_env.Extend(closest_config, new_config, epsilon)

                if new_config is not None:
                    
                    new_id = tree.AddVertex(new_config, C_vertex)
                    tree.AddEdge(closest_id, new_id)
                    self.planning_env.PlotEdge_RRT(new_config, closest_config) # PLOT

                    closest_goal_id, closest_goal_config, closest_goal_cost = tree.GetNearestVertex(goal_config)
                    goalDist = self.planning_env.ComputeDistance_RRT(closest_goal_config, goal_config)

                    # raw_input('...')

        goal_id = tree.AddVertex(goal_config, 0)
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
