import random

from agent import *

class Node:
    def __init__(self, state, parent) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        

class MonteCarloPlanner:

    def __init__(self, world, agent : Agent) -> None:
        self.world = world
        self.agent = agent
        self.root = Node(agent.pos, None)

    def plan(self, iterations=100):

        # print("Planning...")

        # perform 100 iterations of MCTS
        for i in range(iterations):
            self.mcts_iteration(self.root, 0)

        # get best child of root node
        ucb_values = [self.compute_ucb_value(child) for child in self.root.children]
        index = ucb_values.index(max(ucb_values))
        best_node = self.root.children[index]

        self.root = best_node

        # return best child's state
        return best_node.state, ucb_values

    def mcts_iteration(self, node, depth):

        # if node.parent is not None:
        #     print("Iteration: ", depth, node.parent.state, node.state, node.visits, node.value, len(node.children))
        # else:
        #     print("Root Iteration: ", depth, node.state, node.visits, node.value, len(node.children))
        
        # check if node is a leaf node
        if node.visits == 0:

            # expand node based on available actions
            child_node = self.expand(node)

            # perform random simulation from child node to get score
            score = self.simulate(child_node)

            # print("Computed Simulation Score: ", score)

            # adjust value for ancestor nodes upto the root
            child_node.value = score
            self.backpropagate(node, score)
            
        # if node is not a leaf node
        else:

            # print("Non-leaf Node: ", node.state, node.visits, node.value, len(node.children))

            # compute UCB value for each child
            child_states = [(child.state, child.visits, child.value) for child in node.children]
            # print("\tChild States: ", child_states)

            ucb_values = [self.compute_ucb_value(child) for child in node.children]

            # print("UCB Values: ", ucb_values)

            # select child with highest UCB value
            index = ucb_values.index(max(ucb_values))

            # find best child
            best_node = node.children[index]

            # recurse
            self.mcts_iteration(best_node, depth + 1)

    
    def get_next_available_actions(self, node):
        # append all tuples of (delta_x, delta_y) that don't cause the state to hit obstacles or go out of bounds
        
        available_actions = []

        for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if self.check_action_obstacles(node.state, action, self.world.obstacles):
                if self.check_action_boundaries(node.state, action, self.world.grid_width):
                    available_actions.append(action)

        return available_actions
        

    def compute_action_result(self, state, action):
        position = state + action
        return position


    def expand(self, node):

        # get all available actions from current state
        available_actions = self.get_next_available_actions(node)

        for action in available_actions:
            
            new_state = self.compute_action_result(node.state, action)
            post_node = Node(new_state, node)

            node.children.append(post_node)

        new_node = random.choice(node.children)

        return new_node

    def compute_ucb_value(self, node : Node):
        # compute UCB value for node
        exploration_weight = 2.0

        if node.visits == 0:
            return float('inf')
        
        exploitation_score = node.value / node.visits
        exploration_term = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
        
        return exploitation_score + exploration_term
    
    def simulate(self, node):

        # perform random simulation from child node to get score
        score = 0

        random_state = node.state

        for i in range(100):
            available_actions = self.get_next_available_actions(node)
            random_action = random.choice(available_actions)
            random_state = self.compute_action_result(random_state, random_action)
            score -= 1

            if not(self.check_action_obstacles(random_state, random_action, self.world.obstacles)):
                score -= 5000
                break

            elif not(self.check_action_boundaries(random_state, random_action, self.world.grid_height)):
                score -= 5000
                break

            elif np.linalg.norm(random_state - self.world.goal) < self.world.goal_margin:
                score += 10000
                break

        return score
    
    def backpropagate(self, node, score):

        # print("Backpropagating: ", node.state, score)

        # adjust value for ancestor nodes upto the root
        node.visits += 1
        node.value += score

        if node.parent is not None:
            self.backpropagate(node.parent, score)

    def check_action_obstacles(self, state, action, obstacles):

        position = state + action

        collision = False
        for obstacle in obstacles:

            obstacle_size = obstacle[2]
            obstacle_min_x = obstacle[0] - obstacle_size
            obstacle_max_x = obstacle[0] + obstacle_size
            obstacle_min_y = obstacle[1] - obstacle_size
            obstacle_max_y = obstacle[1] + obstacle_size

            if position[0] > obstacle_min_x and position[0] < obstacle_max_x:
                if position[1] > obstacle_min_y and position[1] < obstacle_max_y:
                    collision = True
                    break

        return not(collision)
                
    def check_action_boundaries(self, state, action, grid_size):

        position = state + action

        if position[0] < 0 or position[0] > grid_size:
            return False
        if position[1] < 0 or position[1] > grid_size:
            return False

        return True