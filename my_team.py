 baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from contest.capture import GameState

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point, manhattan_distance


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLearningAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########








class QlearningAgent(CaptureAgent):

    def init(self, index, time_for_computing=.1, alpha = 0.2, epsilon = 0.05, discount = 0.8):
        super().init(index, time_for_computing)
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.q_table = util.Counter()
        self.start = None
        self.index = index

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def getQValue(self, state, action):
        return self.q_table[state, action]

    def computeValueFromQValues(self, state):
        actions = state.get_legal_actions(self.index)
        maximumq = max(self.getQValue(state, action) for action in actions) if actions else 0

        return maximumq

    def computeActionFromQValues(self, state):
        actions = state.get_legal_actions(self.index)
        best_qvalue = self.getValue(state)
        action = random.choice([action for action in actions if self.getQValue(state, action) == best_qvalue]) if actions else None



        return action

    def getAction(self, state):


        legalActions = state.get_legal_actions(self.index)

        if legalActions:
            action = random.choice(legalActions) if util.flip_coin(self.epsilon) else self.getPolicy(state)
        else: action = None

        return action

    def update(self, state, action, nextState, reward):

        updated_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState))
       #print("dit his den waarde voor aanpassing: ",  self.q_table[state, action] )
        self.q_table[state, action] = updated_value


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class OffensiveQLearningAgent(QlearningAgent):

    def init(self, index, time_for_computing=.1, alpha=0.2, epsilon=0.05, discount=0.8):
        super().init(index, time_for_computing, alpha, epsilon, discount)
        #variabel om terug te keren naar eigen kant als het te gevaarlijk wordt
        self.returning = False

    def getReward(self, game_state, next_state):
        reward = -1  # basisstap straf om niet te blijven rondlopen

        current_pos = game_state.get_agent_position(self.index)
        next_pos = next_state.get_agent_position(self.index)

        #kijken hoever we zijn van ghosts en of we tegen ghosts botsen in de volgende state
        opponents_indices = self.get_opponents(game_state)
        opponents_states = [game_state.get_agent_state(i) for i in opponents_indices]
        opponents_positions = [enemy.get_position() for enemy in opponents_states
                               if not enemy.is_pacman and enemy.get_position() is not None]

        for ghost_pos in opponents_positions:
            dist = self.get_maze_distance(next_pos, ghost_pos)
            if dist == 0:
                reward -= 1000  # we botsen => grote straf
            elif dist < 3:
                reward -= 100 / dist  # hoe dichterbij, hoe hoger de straf

        # Beloning voor het eten van voedsel
        food_list = self.get_food(game_state).as_list()
        if next_pos in food_list:
            reward += 5

        # Beloning als het food terugbrengt naar zijn kant
        current_score = self.get_score(game_state)
        next_score = self.get_score(next_state)
        if next_score > current_score:
            brought_food = next_score - current_score #sinds 1 food 1 punt geeft kunnen we zo het aantal opgegeten food ophalen (andere optie mss later een counter bijhouden)
            if brought_food < 5: #straf voor niet genoeg teruggebracht
                reward -= 10
            else:
                reward += (next_score - current_score) * 10 # beloning afhankelijk van hoeveel het teruggebracht heeft

        #om de agent te aanmoedigen om richting het eten te bewegen
        if food_list:
            current_food_distance = min([self.get_maze_distance(current_pos, food) for food in food_list])
            next_food_distance = min([self.get_maze_distance(next_pos, food) for food in food_list])
            if next_food_distance < current_food_distance: #we liggen dichterbij het eten
                reward += (- next_food_distance + current_food_distance) /2 # we belonen voor dichterbij eten gaan
            else:
                reward -= (next_food_distance - current_food_distance) / 2 # we straffen voor het verder weg gaan voor het eten

        return reward

    def choose_action(self, game_state):

        #in de choose action methode sturen we keuze van acties aan de hand van extra informatie om sneller goede acties te kiezen

        #we halen de legale acties op
        legalActions = game_state.get_legal_actions(self.index)
        #mogelijke verbetering voor later (probeer te checken of stop acties)

        current_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()

        # we gaan op zoek naar dichtsbijzijnde eten
        nearest_food = None
        if food_list:
            #zoek minimale afstand tot eten
            min_distance = min([self.get_maze_distance(current_pos, food) for food in food_list])

            #zoek eten dat overeenkomt met afstand, deze wordt dan een soort van goal voor ons agent
            nearest_food = [food for food in food_list if self.get_maze_distance(current_pos, food) == min_distance][0]



        # Controle voor ghosts
        opponents_indices = self.get_opponents(game_state)
        opponents_states = [game_state.get_agent_state(i) for i in opponents_indices]
        ghosts = [ghost for ghost in opponents_states if not ghost.is_pacman and ghost.get_position() is not None]
        if ghosts:
            ghost_dist = min([self.get_maze_distance(current_pos, ghost.get_position()) for ghost in ghosts])
            if ghost_dist <= 2: #als een ghost zich te dicht bij ons bevindt, gaan we terug om de punten die we al hebben te verzekeren
                self.returning = True

        best_action = None
        best_value = float('-inf')

        #
        for action in legalActions:
            next_state = game_state.generate_successor(self.index, action)
            next_pos = next_state.get_agent_position(self.index)

            if self.returning:
                # Als er gevaar is, ga dan richting thuis
                start_pos = self.start
                dist = self.get_maze_distance(next_pos, start_pos)
                action_value = 1/dist  # hoe kleiner de afstand, hoe beter
            elif nearest_food: #veilige afstand van ghosts
                # Anders, ga richting het eten
                dist = self.get_maze_distance(next_pos, nearest_food)
                action_value = 1/dist
            else:
                action_value = 0

            #action value geeft een soort van richtlijn in vroege stadium van q learning traning
            action_value += self.getQValue(game_state, action)

            if action_value > best_value:
                best_value = action_value
                best_action = action

        # exploratie vs exploitatie
        if util.flip_coin(self.epsilon):
            best_action = random.choice(legalActions)

        #we updaten de q-table (exploratie
        next_state = game_state.generate_successor(self.index, best_action)
        reward = self.getReward(game_state, next_state)
        self.update(game_state, best_action, next_state, reward)

        #check of we de return flag moeten aanpassen
        if self.returning:
            if self.is_own_territory(current_pos, game_state):
                self.returning = False

        return best_action

    def is_own_territory(self, position, game_state):
        #hier gaan we ervan uit dat rechterkant = rode team en linkerkant = blauwe team
        red = game_state.is_on_red_team(self.index)
        mid = game_state.get_walls().width // 2
        if red:
            return position[0] < mid
        else:
            return position[0] >= mid

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def init(self, index, time_for_computing=.1):
        super().init(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
      #  print("piep")
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 18:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):# wat heb je allemaal nodig
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists) if my_state.scared_timer == 0 else -10000

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        #bewaken aan de grens:

        opps_index = self.get_opponents(game_state)
        opps_states = [game_state.get_agent_state(ind) for ind in opps_index]
        opps_pos = [op.get_position() for op in opps_states if not op.is_pacman and op.get_position() is not None]

        if opps_pos:
            distances = [self.get_maze_distance(my_pos,opp) for opp in opps_pos]
            minimal_distance = min(distances)
            if minimal_distance < 4:
                features['future_pacman'] = 11/ minimal_distance if minimal_distance > 0 else 11/ minimal_distance + 0.1
            else:
                features['future_pacman'] = 0
        else:
            features['future_pacman'] = 0


        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'future_pacman': 500}