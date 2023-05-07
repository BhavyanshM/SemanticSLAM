import random

from agent import *

class Node:
    def __init__(self, state) -> None:
        self.state = state
        self.parent = None
        self.children = []
        self.visits = 0
        self.value = 0
        

class MonteCarloPlanner:

    def __init__(self, world, agent : Agent) -> None:
        self.world = world
        self.agent = agent
        self.root = Node(agent.pos)

    def plan(self):

        # perform 100 iterations of MCTS
        for i in range(100):
            self.mcts_iteration(self.root, 0)

        # get best child of root node
        ucb_values = [self.compute_ucb_value(child) for child in self.root.children]
        index = ucb_values.index(max(ucb_values))
        best_node = self.root.children[index]

        self.root = best_node

        # return best child's state
        return best_node.state

    def mcts_iteration(self, node, depth):
        
        # check if node is a leaf node
        if node.visits == 0:

            # expand node based on available actions
            child_node = self.expand(node)

            # perform random simulation from child node to get score
            score = self.simulate(child_node)

            # adjust value for ancestor nodes upto the root
            self.backpropagate(child_node, score)
            
        # if node is not a leaf node
        else:

            # compute UCB value for each child
            ucb_values = [self.compute_ucb_value(child) for child in node.children]

            # select child with highest UCB value
            index = ucb_values.index(max(ucb_values))

            # find best child
            best_node = node.children[index]

            # recurse
            self.mcts_iteration(best_node, depth + 1)

    
    def get_next_available_actions(self):
        # append all tuples of (delta_x, delta_y) that don't cause the state to hit obstacles or go out of bounds
        
        available_actions = []

        for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if self.agent.check_action_obstacles(action, self.world.obstacles):
                available_actions.append(action)

        return available_actions
        

    def expand(self, node):
        # get all available actions from current state
        available_actions = self.get_next_available_actions()

        action = random.choice(available_actions)
        
        new_state = self.agent.apply_action(action)
        new_node = Node(new_state)
        new_node.parent = node

        node.children.append(new_node)

        return new_node

    def compute_ucb_value(self, node : Node):
        # compute UCB value for node
        exploration_weight = 2.0

        exploitation_score = node.value / node.visits
        exploration_term = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
        
        return exploitation_score + exploration_term
    
    def simulate(self, node):

        # perform random simulation from child node to get score
        score = 0

        for i in range(100):
            self.agent.random_update(self.world.grid_height)
            if self.agent.check_action_obstacles(self.agent.pos, self.world.obstacles):
                score -= 1

            elif not(self.agent.check_action_boundaries(self.agent.pos, self.world.grid_height)):
                score -= 1

            elif np.linalg.norm(self.agent.pos - self.world.goal) < 30:
                score += 1000
                break

        return score
    
    def backpropagate(self, node, score):
        # adjust value for ancestor nodes upto the root
        node.visits += 1
        node.value += score

        if node.parent is not None:
            self.backpropagate(node.parent, score)