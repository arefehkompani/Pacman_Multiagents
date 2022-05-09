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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newGhostStates = successorGameState.getGhostStates()
        food_list = successorGameState.getFood().asList()
        newFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = 0

        # Check if the next state is winning/losing
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # checks to see if pacman ate some food
        pacX, pacY = newPos
        if newFood[pacX][pacY]:
            score += 1

        food_distances = {}
        if food_list:
            for food in food_list:
                food_distances[food] = manhattanDistance(food, newPos)

            closest_food = min(food_distances, key=food_distances.get)
            # for ghost in newGhostStates:
            if newScaredTimes[0] > 0:
                score += 2 / food_distances[closest_food]
                food_list.remove(closest_food)
            else:
                score += 1 / food_distances[closest_food]
                food_list.remove(closest_food)

        # distance from ghost
        if newGhostStates:
            for ghost in newGhostStates:
                ghost_distance = manhattanDistance(newPos,
                                                   ghost.getPosition())

                # if ghost is close return large negative value
                if ghost_distance <= 1:
                    return float('-inf')

                # otherwise subtract the inverse (closer -> larger negative)
                score -= 1 / ghost_distance

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        # Total number of agents (including pacman)
        self.num_agents = gameState.getNumAgents()

        # create a list with the ID of all of the ghosts
        self.ghosts = []
        for i in range(self.num_agents - 1):
            self.ghosts.append(i + 1)

        depth = 0                   # current depth
        best_action = None          # best course of action
        best_value = float('-inf')  # score for an evaluated action

        # all the legal actions pacman can currently take
        pacman_actions = gameState.getLegalActions(0)

        # Iterate through all of possible actions
        for action in pacman_actions:
            # future game state after pacman action taken
            pacman_future = gameState.generateSuccessor(0, action)
            # return from applying min node for a ghost agent on pacman actions
            value = self.minimize_action(pacman_future, self.ghosts[0], depth)

            # Pacman takes the best value, cause its a max node
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def maximize_action(self, gameState, depth):
        """
            Apply max node implementation
        """

        # Legal pacman actions
        pacman_actions = gameState.getLegalActions(0)

        # see if we have reached the depth limit, return value of state
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # if there are not actions left return value of state
        if not pacman_actions:
            return self.evaluationFunction(gameState)

        # store values for each node for a given pacman action
        values = []
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            # for a given gamestate min value for each
            # state resulting from said action
            values.append(
                self.minimize_action(pacman_future, self.ghosts[0], depth))

        # for all the nodes return the largest value
        return max(values)

    def minimize_action(self, gameState, ghost_ID, depth):

        """ For the MIN Players or Agents  """

        # gets the legal actions for ghost
        ghost_actions = gameState.getLegalActions(ghost_ID)

        # see if we have reached the depth limit, return value of state
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # if there are not actions left return value of state
        if not ghost_actions:
            return self.evaluationFunction(gameState)

        min_reward = float('inf')
        values = []

        # check we aren't the last ghost
        if ghost_ID < gameState.getNumAgents() - 1:
            for action in ghost_actions:
                # future caused by ghost taking its action
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                # every action of ghost X needs to be evaluated by ghost X+1
                # until run out of ghosts
                values.append(
                    self.minimize_action(ghost_future, ghost_ID + 1, depth))
        else:
            # last ghost need to call maximize afterwards
            for action in ghost_actions:
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                values.append(self.maximize_action(ghost_future, depth + 1))

        # make sure not empty list we are calling min on
        if not values:
            return min_reward
        else:
            return min(values)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimize_action(gameState, ghost, d, alpha, beta):
          val = float('inf')

          for action in gameState.getLegalActions(ghost):
            tempState =gameState.generateSuccessor(ghost, action)
            tempVal, action = abPruning(tempState, ghost +1, d, alpha, beta)
            if tempVal < val:
              val = tempVal
            if val < alpha:
                return val
            beta = min(beta, tempVal)

          return val

        def maximize_action(gameState, d, alpha, beta):
          val = float('-inf')
          #default best move is stop
          bestAction = 'Stop'

          for action in gameState.getLegalActions(0):
            tempState =gameState.generateSuccessor(0, action)
            tempVal, tempAction = abPruning(tempState, 1, d, alpha, beta)
            if tempVal > val:
              val = tempVal
              bestAction = action
            if val > beta:
                return (val, action)
            alpha = max(alpha, tempVal)
          return (val, bestAction)

        #depending on whose move and depth, either maximise or minimise
        def abPruning(gameState, agent, d, alpha, beta):
          #each player gets on move for each depth
          if agent >= gameState.getNumAgents():
            agent = 0
            d += 1
          #return eval fn when game finished or depth reached
          if (gameState.isWin() or gameState.isLose() or self.depth < d):
            return (self.evaluationFunction(gameState), '')
          #pacman's move gets max value, the ghosts get min value          
          if 0 == agent:
            return maximize_action(gameState, d, alpha, beta)
          else:
            return (minimize_action(gameState, agent, d, alpha, beta), '')

        #first depth = 1, first agent = pacman (zero)
        d = 1
        firstAgent = 0
        value, action = abPruning(gameState, firstAgent, d, float('-inf'), float('inf'))
        return action


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
        def expectiValue(gameState, ghost, d):
          val = 0
          legalActions = gameState.getLegalActions(ghost)

          #sum all values
          for action in legalActions:
            tempState =gameState.generateSuccessor(ghost, action)
            tempVal, action = expectimaxDecision(tempState, ghost +1, d)
            val += tempVal

          #return average utility value
          return val / len(legalActions)

        def maximize_action(gameState, d):
          val = float('-inf')
          #default best move is stop
          bestAction = 'Stop'

          for action in gameState.getLegalActions(0):
            tempState =gameState.generateSuccessor(0, action)
            tempVal, tempAction = expectimaxDecision(tempState, 1, d)
            if tempVal > val:
              val = tempVal
              bestAction = action

          return (val, bestAction)

        #depending on whose move and depth, either maximise or minimise
        def expectimaxDecision(gameState, agent, d):
          #each player gets on move for each depth
          if agent >= gameState.getNumAgents():
            agent = 0
            d += 1
          #return eval fn when game finished or depth reached
          if (gameState.isWin() or gameState.isLose() or self.depth < d):
            return (self.evaluationFunction(gameState), '')
          #pacman's move gets max value, the ghosts get min value          
          if 0 == agent:
            return maximize_action(gameState, d)
          else:
            return (expectiValue(gameState, agent, d), '')

        #first depth = 1, first agent = pacman (zero)
        d = 1
        firstAgent = 0
        value, action = expectimaxDecision(gameState, firstAgent, d)
        return action
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    posPacman = currentGameState.getPacmanPosition() #pacman position
    stateFantasmas = currentGameState.getGhostStates() #State in which the ghosts are found
    posFood = currentGameState.getFood().asList() #List of the position of each of the foods
    score = currentGameState.getScore() #Score
    numFood = currentGameState.getNumFood() #Number of foods left
    
    disFood = [manhattanDistance(Food, posPacman) for Food in posFood] #Manhattan list between each food and the position of the pacman

    #It checks if there are food items on the map and considers the shortest distance of the food items as their weight

    if len(disFood):
        minDisFood = min(disFood)
    else:
        minDisFood = 0

    #The position of each of the ghosts is reviewed and a weight is given to them at the moment in which the pacman is allowed to eat them
    scoreGhost = 0

    for ghost in stateFantasmas:
        disGhost = manhattanDistance(ghost.getPosition(), posPacman)
        if ghost.scaredTimer > disGhost:
            scoreGhost = scoreGhost - disGhost + 50
    
    return score - minDisFood - numFood + scoreGhost
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
