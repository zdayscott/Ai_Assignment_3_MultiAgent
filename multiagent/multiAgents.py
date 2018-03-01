# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #check if win
        if successorGameState.isWin():
            return float("inf")
        #Find ghost
        ghostPos = successorGameState.getGhostPosition(1)
        #Set a base score from distance from ghost, Farther = best, Close = worst
        score = util.manhattanDistance(ghostPos, newPos) + successorGameState.getScore()
        #Find closest food
        foodList = newFood.asList()

        closestFood = 1000

        for food in foodList:
            if(manhattanDistance(newPos,food) < closestFood):
                closestFood = manhattanDistance(newPos, food)

        score -= closestFood
        #If this move collects a food add to score
        if(currentGameState.getNumFood() > successorGameState.getNumFood()):
            score += 50
        
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
        """
        "*** YOUR CODE HERE ***"
        print self.depth

        #returns the value best suited for pacman
        def maxValue(state, depth, ghosts):
            #test prints
            print depth
            print "Pac"
            #if depth is 0 return estimated value
            if state.isWin() or state.isLose() or depth == 0:
                print ("Pac finish")
                return self.evaluationFunction(state)

            v = -(float("inf"))
            moves = gameState.getLegalActions(0)

            for move in moves:
                next = gameState.generateSuccessor(0, move)
                v = max(v, minValue(next, depth, 1, ghosts))

            return v

        #returns the value worst suited for pacman iterated over all ghost agents
        def minValue(state, depth, ghostindex, ghosts):
            #test prints
            print depth
            print "Ghost"
            #if depth is 0 return estimated value
            if state.isWin() or state.isLose() or depth == 0:
                print("Ghost finish")
                return self.evaluationFunction(state)
            moves = gameState.getLegalActions(ghostindex)
            #Initialize the value variable
            v = float("inf")

            #if last ghost
            if ghostindex == ghosts:
                #iterate over all valid moves
                for move in moves:
                    #increase depth by 1 level (indexed by decreasing to zero)
                    v = min(v, maxValue(gameState.generateSuccessor(ghostindex, move), depth - 1, ghosts))
            #if not last ghost
            else:
                #iterate over all valid moves
                for move in moves:
                    v = min(v, minValue(gameState.generateSuccessor(ghostindex, move), depth, ghostindex + 1, ghosts))

            return v

        #Establish Agents and Legal Actions available to Pacman
        totalGhosts = gameState.getNumAgents() - 1
        pactions = gameState.getLegalActions(0)

        #print totalGhosts

        #Generate Action Tree
        bestmove = Directions.STOP
        bestvalue = -(float("inf"))
        for action in pactions:
            next = gameState.generateSuccessor(0, action)
            v = minValue(next, self.depth+1, 1, totalGhosts)
            if bestvalue < v:
                bestvalue = v
                bestmove = action

        return bestmove


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

