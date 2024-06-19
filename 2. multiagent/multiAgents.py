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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        pain = 0
        ghost_stuff = 0
        food_stuff = 0

        food_now = currentGameState.getFood().asList()
        mh_food = []
        for foody in newFood.asList():
            mh_food.append(manhattanDistance(newPos, foody))
        if len(mh_food) == 0:
            return 69


        ghosts = [ghost.getPosition() for ghost in newGhostStates]
        mh_ghost = []
        for ghosty in ghosts:
            mh_ghost.append(manhattanDistance(newPos, ghosty))
        if 0 in mh_ghost:
            return -69


        if len(newFood.asList()) < len(food_now):
            pain += 10
        if currentGameState.getPacmanPosition() == newPos:
            pain -= 10

        if min(mh_ghost) < 1:
            food_stuff += 1 * (1 / (min(mh_food) + 3))
            for ghost in mh_ghost:
                ghost_stuff -= 15 * (1 / ((ghost + 3) ** 2))
        elif (min(mh_ghost) >= 1) & (min(mh_ghost) < 3):
            food_stuff += 5 * (1 / (min(mh_food) + 3))
            for ghost in mh_ghost:
                ghost_stuff -= 5 * (1 / ((ghost + 3) ** 2))       
        else:
            food_stuff += 10 * (1 / (min(mh_food) + 3))
            for ghost in mh_ghost:
                ghost_stuff -= 1 * (1 / ((ghost + 3) ** 2))
        
        return pain + ghost_stuff + food_stuff

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def get_value(gameState, depth, index):
            if gameState.isWin() or gameState.isLose() or (self.depth == depth):
                return self.evaluationFunction(gameState)
            if index == 0:
                return maxi(gameState, depth, index)
            else:
                return mini(gameState, depth, index)
    
        def mini(gameState, depth, index):
            sucs = [gameState.generateSuccessor(index, action) for action in gameState.getLegalActions(index)]
            smallest = float('inf')
            for suc in sucs:
                if index == (gameState.getNumAgents() - 1):
                    new = get_value(suc, depth + 1, 0)
                    if new < smallest:
                        smallest = new
                else:
                    new = get_value(suc, depth, index + 1)
                    if new < smallest:
                        smallest = new
            return smallest

        def maxi(gameState, depth, index):
            sucs = [gameState.generateSuccessor(index, action) for action in gameState.getLegalActions(index)]
            biggest = float('-inf')
            for suc in sucs:
                new = get_value(suc, depth, 1)
                if new > biggest:
                    biggest = new
            return biggest
        
        successors = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        vals = [get_value(suc, 0, 1) for suc in successors]
        return gameState.getLegalActions(0)[vals.index(max(vals))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def get_value(gameState, depth, index, alpha, beta):
            if gameState.isWin() or gameState.isLose() or (self.depth == depth):
                return self.evaluationFunction(gameState)
            if index == 0:
                return maxi(gameState, depth, index, alpha, beta)
            else:
                return mini(gameState, depth, index, alpha, beta)
    
        def mini(gameState, depth, index, alpha, beta):
            smallest = float('inf')
            
            for action in gameState.getLegalActions(index):
                suc = gameState.generateSuccessor(index, action)
                if index == (gameState.getNumAgents() - 1):
                    new = get_value(suc, depth + 1, 0, alpha, beta)
                else:
                    new = get_value(suc, depth, index + 1, alpha, beta)
                if new < smallest:
                    smallest = new
                if smallest < alpha:
                    return new
                beta = min(beta, smallest)
            return smallest

        def maxi(gameState, depth, index, alpha, beta):
            biggest = float('-inf')
            for action in gameState.getLegalActions(index):
                suc = gameState.generateSuccessor(index, action)
                new = get_value(suc, depth, 1, alpha, beta)
                if new > biggest:
                    biggest = new
                if biggest > beta:
                    return new
                alpha = max(alpha, biggest)
            return biggest
        
        alpha = float('-inf')
        beta = float('inf')
        successors = []
        best = 0

        successors = []
        for action in gameState.getLegalActions(0):
            successors.append(gameState.generateSuccessor(0, action)) 
        for suc in successors:
            if get_value(suc, 0, 1, alpha, beta) > alpha:
                alpha = get_value(suc, 0, 1, alpha, beta)
                best = successors.index(suc)
        return gameState.getLegalActions(0)[best]

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
        def get_value(gameState, depth, index):
            if gameState.isWin() or gameState.isLose() or (self.depth == depth):
                return self.evaluationFunction(gameState)
            if index == 0:
                return maxi(gameState, depth, index)
            else:
                return average(gameState, depth, index)
    
        def average(gameState, depth, index):
            avg = 0
            probability = 1 / len(gameState.getLegalActions(index))
            
            for action in gameState.getLegalActions(index):
                if index == gameState.getNumAgents() - 1:
                    avg += probability * get_value(gameState.generateSuccessor(index, action), depth + 1, 0)
                else:
                    avg += probability * get_value(gameState.generateSuccessor(index, action), depth, index + 1)
            return avg

        def maxi(gameState, depth, index):
            sucs = [gameState.generateSuccessor(index, action) for action in gameState.getLegalActions(index)]
            values = []
            for suc in sucs:
                values.append(get_value(suc, depth, 1))
            return max(values)
        
        successors = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        vals = [get_value(suc, 0, 1) for suc in successors]
        return gameState.getLegalActions(0)[vals.index(max(vals))]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I basically went primitive mode and just adjusted weights most of the time
    """
    

    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    pellets = currentGameState.getCapsules()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    currScore = currentGameState.getScore()

    # print(currScore)

    pain = 0
    ghost_stuff = 0
    food_stuff = 0

    food_lst = Food.asList()
    mh_food = []
    for foody in food_lst:
        mh_food.append(manhattanDistance(Pos, foody))
    if len(mh_food) == 0:
        return 1000

    ghosts = [ghost.getPosition() for ghost in GhostStates]
    mh_ghost = []
    for ghosty in ghosts:
        mh_ghost.append(manhattanDistance(Pos, ghosty))
    if 0 in mh_ghost:
        return -1000

    # if min(mh_ghost) <= 1:
    #     food_stuff += 1 * (1 / (min(mh_food)))
    #     for ghost in mh_ghost:
    #         ghost_stuff -= 15 * (1 / ((ghost) ** 2))
    # elif (min(mh_ghost) > 1) & (min(mh_ghost) < 3):
    #     food_stuff += 5 * (1 / (min(mh_food)))
    #     for ghost in mh_ghost:
    #         ghost_stuff -= 5 * (1 / ((ghost) ** 2))       
    # else:
    #     food_stuff += 20 * (1 / (min(mh_food)))
    #     for ghost in mh_ghost:
    #         ghost_stuff -= 5 * (1 / ((ghost) ** 2))
    # if sum(ScaredTimes) > 0:
    #     food_stuff += 100 * (1 / (min(mh_food) + 3))

    food_stuff = 20 / min(mh_food)
    ghost_stuff = -2 / min(mh_ghost)
    pellet_stuff = 4 * len(pellets)
    # food_stuff += 10 / min(mh_food)

    pain = -8 * len(food_lst)

    # print(pain + ghost_stuff + food_stuff + pellet_stuff + currScore)

    return pain + ghost_stuff + food_stuff + pellet_stuff + currScore

# Abbreviation
better = betterEvaluationFunction
