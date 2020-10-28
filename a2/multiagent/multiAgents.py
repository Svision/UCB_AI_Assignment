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
from math import inf, sqrt

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(successorGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)
        score = 0
        # Check lose
        if successorGameState.isLose():
            return -inf

        # Check ghost
        min_ghost_dis = inf
        for ghost in newGhostStates:
            ghost_dis = manhattanDistance(ghost.getPosition(), newPos)
            if ghost_dis < min_ghost_dis:
                min_ghost_dis = ghost_dis
        if min_ghost_dis < 2 and action == 'Stop':
            return -inf
        score += sqrt(min_ghost_dis)

        # Check Food
        min_food_dis = inf
        newFood = newFood.asList()
        currFood = currFood.asList()
        for food_pos in newFood:
            food_dis = manhattanDistance(food_pos, newPos)
            if food_dis < min_food_dis:
                min_food_dis = food_dis
        score -= min_food_dis
        if newPos in currFood:
            return inf

        # Check Scare time
        scare_time = 0
        for t in newScaredTimes:
            scare_time += t
        score += scare_time

        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        numAgents = gameState.getNumAgents()

        def DFMiniMax(state, depth=self.depth, curr=-1):
            best_move = None
            curr += 1
            agentIndex = curr % numAgents
            if depth*numAgents == curr or state.isLose() or state.isWin():
                return "Stop", self.evaluationFunction(state)
            if agentIndex == 0:
                # Pacman
                value = -inf
            else:
                # Ghost
                value = inf
            for move in state.getLegalActions(agentIndex):
                nxt_state = state.generateSuccessor(agentIndex, move)
                nxt_move, nxt_value = DFMiniMax(nxt_state, depth, curr)
                if agentIndex == 0 and value < nxt_value:
                    value, best_move = nxt_value, move
                if agentIndex >= 1 and value > nxt_value:
                    value, best_move = nxt_value, move
            return best_move, value

        return DFMiniMax(gameState)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def DFMiniMax(state, curr=-1):
            best_move = None
            curr += 1
            agentIndex = curr % numAgents
            if self.depth * numAgents == curr or state.isLose() or state.isWin():
                return "Stop", self.evaluationFunction(state)
            if agentIndex == 0:
                # Pacman
                value = -inf
            else:
                # Ghost
                value = inf
            for move in state.getLegalActions(agentIndex):
                nxt_state = state.generateSuccessor(agentIndex, move)
                nxt_move, nxt_value = DFMiniMax(nxt_state, curr)
                if agentIndex == 0 and value < nxt_value:
                    value, best_move = nxt_value, move
                if agentIndex >= 1 and value > nxt_value:
                    value, best_move = nxt_value, move
            return best_move, value

        return DFMiniMax(gameState)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    # Check terminal
    if currentGameState.isLose():
        return -inf
    if currentGameState.isWin():
        return inf

    # Check Wall
    if currentGameState.hasWall(pos[0], pos[1]):
        return -inf

    # Check ghost
    min_ghost_dis = inf
    min_scared_ghost_dis = inf
    for ghost in ghostStates:
        ghost_dis = manhattanDistance(ghost.getPosition(), pos)
        if not ghost.scaredTimer:
            if ghost_dis < min_ghost_dis:
                min_ghost_dis = ghost_dis
        else:
            if ghost_dis < min_scared_ghost_dis:
                min_scared_ghost_dis = ghost_dis
    if min_ghost_dis < 2:
        return -inf
    if min_scared_ghost_dis == inf:
        min_scared_ghost_dis = 0
    score += sqrt(min_ghost_dis)
    score += min_scared_ghost_dis

    # Check Food
    min_food_dis = inf
    max_food_dis = -inf
    food = food.asList()
    for food_pos in food:
        food_dis = manhattanDistance(food_pos, pos)
        if food_dis < min_food_dis:
            min_food_dis = food_dis
        if food_dis > max_food_dis:
            max_food_dis = food_dis
    score -= min_food_dis
    score += sqrt(max_food_dis)

    # Check Capsule
    score -= len(capsules)

    # Check Scare time
    scare_time = 0
    for t in scaredTimes:
        scare_time += t
    score += scare_time

    return score

# Abbreviation
better = betterEvaluationFunction
