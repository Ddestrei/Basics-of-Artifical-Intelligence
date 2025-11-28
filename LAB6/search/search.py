# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def dfs_impl(state, problem, way, path, states):
    if path is not None: way.append(path)
    states.append(state)
    #print(way)
    if problem.isGoalState(state):
        return way
    successors = problem.getSuccessors(state)
    return_v = None
    for s in successors:
        if s[0] in states: continue
        return_v = dfs_impl(s[0], problem, way.copy(), s[1], states)
        if return_v is not None:
            return return_v

    return None

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    start_sate = problem.getStartState()
    way = []
    return dfs_impl(start_sate, problem, way,None, [])

def dfs_impl_D(state, problem, way, path, states, depth):
    if depth == 0:
        return None
    if path is not None: way.append(path)
    states.append(state)
    #print(way)
    if problem.isGoalState(state):
        return way
    successors = problem.getSuccessors(state)
    return_v = None
    for s in successors:
        if s[0] in states: continue
        return_v = dfs_impl_D(s[0], problem, way.copy(), s[1], states, depth - 1)
        if return_v is not None:
            return return_v




def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    states = []
    paths = []
    index = 0
    curr_state = problem.getStartState()
    states.append(curr_state)
    path = []
    goal = None
    while index < len(states):
        if goal == curr_state:
            break
        successors = problem.getSuccessors(curr_state)
        for s in successors:
            if problem.isGoalState(s[0]):
                goal = s[0]
            if s[0] in states: continue
            states.append(s[0])
            paths.append(s)
        index = index + 1
        if index < len(states):
            curr_state = states[index]

    while goal != 'A':
        for p in paths:
            if p[0] == goal:
                path.append(p[1])
                goal = p[1][2]

    path.reverse()

    return path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
