#!/usr/bin/env python

import argparse, numpy, openravepy, time

from HerbRobot import HerbRobot
from SimpleRobot import SimpleRobot
from HerbEnvironment import HerbEnvironment
from SimpleEnvironment import SimpleEnvironment

from RRT_HerbEnvironment import RRT_HerbEnvironment
from RRT_SimpleEnvironment import RRT_SimpleEnvironment

from AStarPlanner import AStarPlanner
from DepthFirstPlanner import DepthFirstPlanner
from BreadthFirstPlanner import BreadthFirstPlanner
from HeuristicRRTPlanner import HeuristicRRTPlanner

import numpy as np

def main(robot, planning_env, planner):

    # raw_input('Press any key to begin planning')

    start_config = numpy.array(robot.GetCurrentConfiguration())
    if robot.name == 'herb':
        goal_config = numpy.array([ 4.6, -1.76, 0.00, 1.96, -1.15, 0.87, -1.43] )
    else:
        goal_config = numpy.array([3.0, 0.0])

    planning_env.SetGoalParameters(goal_config, 0.05)

    start_time = time.time()
    plan = planner.Plan(start_config, goal_config)
    end_time = time.time()
    print("elapsed time: " + str(end_time - start_time))

    prev_node = plan[0]
    path_length = 0
    for n in plan:
        path_length += np.linalg.norm(prev_node - n)
        prev_node = n.copy()

    print("path length: " + str(path_length))

    traj = robot.ConvertPlanToTrajectory(plan)

    # raw_input('Press any key to execute trajectory')
    # for p in plan:
    #     planning_env.setDOF(p)

    robot_saver = planning_env.robot.CreateRobotStateSaver(
      planning_env.robot.SaveParameters.ActiveDOF
    | planning_env.robot.SaveParameters.ActiveManipulator
    | planning_env.robot.SaveParameters.LinkTransformation)
    with robot_saver:  
        robot.ExecuteTrajectory(traj)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')
    
    parser.add_argument('-r', '--robot', type=str, default='simple',
                        help='The robot to load (herb or simple)')
    parser.add_argument('-p', '--planner', type=str, default='astar',
                        help='The planner to run (astar, bfs, dfs or hrrt)')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Enable visualization of tree growth (only applicable for simple robot)')
    parser.add_argument('--resolution', type=float, default=0.1,
                        help='Set the resolution of the grid (default: 0.1)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('-m', '--manip', type=str, default='right',
                        help='The manipulator to plan with (right or left) - only applicable if robot is of type herb')
    args = parser.parse_args()
    
    openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
    openravepy.misc.InitOpenRAVELogging()

    if args.debug:
        openravepy.RaveSetDebugLevel(openravepy.DebugLevel.Debug)

    env = openravepy.Environment()
    env.SetViewer('qtcoin')
    env.GetViewer().SetName('Homework 2 Viewer')

    # First setup the environment and the robot
    visualize = args.visualize
    if args.robot == 'herb':
        robot = HerbRobot(env, args.manip)
        # if args.planner == 'hrrt':
            # planning_env = RRT_HerbEnvironment(robot)
        # else:
        planning_env = HerbEnvironment(robot, args.resolution)
        visualize = False
    elif args.robot == 'simple':
        robot = SimpleRobot(env)
        # if args.planner == 'hrrt':
            # planning_env = RRT_SimpleEnvironment(robot)
        # else:
        planning_env = SimpleEnvironment(robot, args.resolution)
    else:
        print 'Unknown robot option: %s' % args.robot
        exit(0)

    # Next setup the planner
    if args.planner == 'astar':
        planner = AStarPlanner(planning_env, visualize)
    elif args.planner == 'bfs':
        planner = BreadthFirstPlanner(planning_env, visualize)
    elif args.planner == 'dfs':
        planner = DepthFirstPlanner(planning_env, visualize)
    elif args.planner == 'hrrt':
        planner = HeuristicRRTPlanner(planning_env, visualize)
    else:
        print 'Unknown planner option: %s' % args.planner
        exit(0)

    main(robot, planning_env, planner)

    import IPython
    IPython.embed()

        
    
