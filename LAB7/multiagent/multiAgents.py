# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X   for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFoodList = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        evaluationScore = successorGameState.getScore()

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            scaredTimer = ghostState.scaredTimer
            distance = manhattanDistance(newPos, ghostPos)

            if scaredTimer == 0 and distance > 0:
                if distance <= 2:
                    evaluationScore -= (20.0 / distance) + 500
                elif distance <= 5:
                    evaluationScore -= 10.0 / distance

            elif scaredTimer > 0 and distance > 0:
                if distance < scaredTimer:
                    evaluationScore += 200.0 / distance

        if newFoodList:
            min_food_distance = min([manhattanDistance(newPos, food) for food in newFoodList])

            evaluationScore += 10.0 / min_food_distance

        evaluationScore -= 20 * len(newFoodList)

        capsules_in_successor = len(successorGameState.getCapsules())
        capsules_in_current = len(currentGameState.getCapsules())

        if capsules_in_successor < capsules_in_current:
            evaluationScore += 500

        return evaluationScore


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value, action = self._maxValue(gameState, 0, 0)
        return action

    def _isTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth

    def _getAgentValue(self, gameState, depth, agentIndex):
        if agentIndex == 0:
            return self._maxValue(gameState, depth, agentIndex)
        else:
            return self._minValue(gameState, depth, agentIndex)

    def _maxValue(self, gameState, depth, agentIndex):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        v = -float('inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successor_value, _ = self._getAgentValue(successor, depth, agentIndex + 1)

            if successor_value > v:
                v = successor_value
                bestAction = action

        return (v, bestAction)

    def _minValue(self, gameState, depth, agentIndex):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        v = float('inf')
        bestAction = Directions.STOP
        numAgents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            nextAgentIndex = agentIndex + 1
            nextDepth = depth

            if nextAgentIndex == numAgents:
                nextAgentIndex = 0
                nextDepth = depth + 1

            successor_value, _ = self._getAgentValue(successor, nextDepth, nextAgentIndex)

            if successor_value < v:
                v = successor_value
                bestAction = action

        return (v, bestAction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')
        value, action = self._maxValue(gameState, 0, 0, alpha, beta)
        return action

    def _isTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == self.depth

    def _getAgentValue(self, gameState, depth, agentIndex, alpha, beta):
        if agentIndex == 0:
            return self._maxValue(gameState, depth, agentIndex, alpha, beta)
        else:
            return self._minValue(gameState, depth, agentIndex, alpha, beta)

    def _maxValue(self, gameState, depth, agentIndex, alpha, beta):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        v = -float('inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successor_value, _ = self._getAgentValue(successor, depth, agentIndex + 1, alpha, beta)

            if successor_value > v:
                v = successor_value
                bestAction = action

            if v > beta:
                return (v, bestAction)

            alpha = max(alpha, v)

        return (v, bestAction)

    def _minValue(self, gameState, depth, agentIndex, alpha, beta):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        v = float('inf')
        bestAction = Directions.STOP
        numAgents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            nextAgentIndex = agentIndex + 1
            nextDepth = depth

            if nextAgentIndex == numAgents:
                nextAgentIndex = 0
                nextDepth = depth + 1

            successor_value, _ = self._getAgentValue(successor, nextDepth, nextAgentIndex, alpha, beta)

            if successor_value < v:
                v = successor_value
                bestAction = action

            if v < alpha:
                return (v, bestAction)

            beta = min(beta, v)

        return (v, bestAction)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value, action = self._maxValue(gameState, 0, 0)
        return action

    def _isTerminalState(self, gameState, depth):
        """Checks if the game is over or the max search depth is reached."""
        return gameState.isWin() or gameState.isLose() or depth == self.depth

    def _getAgentValue(self, gameState, depth, agentIndex):
        """Wrapper to call the correct function (MAX or CHANCE) based on the agentIndex."""
        if agentIndex == 0:
            return self._maxValue(gameState, depth, agentIndex)
        else:
            return self._expectedValue(gameState, depth, agentIndex)

    def _maxValue(self, gameState, depth, agentIndex):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        v = -float('inf')
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            successor_value, _ = self._getAgentValue(successor, depth, agentIndex + 1)

            if successor_value > v:
                v = successor_value
                bestAction = action

        return (v, bestAction)

    def _expectedValue(self, gameState, depth, agentIndex):
        if self._isTerminalState(gameState, depth):
            return (self.evaluationFunction(gameState), Directions.STOP)

        legalActions = gameState.getLegalActions(agentIndex)
        numActions = len(legalActions)

        if numActions == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        expected_v = 0.0
        probability = 1.0 / numActions
        numAgents = gameState.getNumAgents()

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)

            nextAgentIndex = agentIndex + 1
            nextDepth = depth

            if nextAgentIndex == numAgents:
                nextAgentIndex = 0
                nextDepth = depth + 1

            successor_value, _ = self._getAgentValue(successor, nextDepth, nextAgentIndex)

            expected_v += successor_value * probability

        return (expected_v, Directions.STOP)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    evaluation = currentGameState.getScore()

    if foodList:
        food_distances = [manhattanDistance(pacmanPos, food) for food in foodList]
        min_food_distance = min(food_distances)
        evaluation += 10.0 / min_food_distance

    evaluation -= 20 * len(foodList)


    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        scaredTimer = ghostState.scaredTimer
        distance = manhattanDistance(pacmanPos, ghostPos)

        if distance == 0:
            if scaredTimer > 0:
                evaluation += 1000
            else:
                return -float('inf')

        elif scaredTimer > 0:
            if distance < scaredTimer:
                evaluation += 200.0 / distance
        else:
            if distance <= 2:
                evaluation -= 500
            elif distance <= 5:
                evaluation -= 10.0 / distance

    evaluation -= 150 * len(capsules)

    # --- 5. Return Final Evaluation ---
    return evaluation


# Abbreviation
better = betterEvaluationFunction
