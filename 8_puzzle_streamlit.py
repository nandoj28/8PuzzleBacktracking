import streamlit as st
import numpy as np
from copy import deepcopy

# Configuration variables
SIZE = 3  # Size of the puzzle
DEPTH = 5  # Maximum depth for backtracking

# Dictionary mapping move directions to their effects
MOVES = {
    'S': (1, 0),  # South: Move down
    'E': (0, 1),  # East: Move right
    'W': (0, -1), # West: Move left
    'N': (-1, 0), # North: Move up
}

def find_blank_position(state):
    """Find the position of the blank space ('*') in the puzzle."""
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            if tile == '*':
                return (i, j)
    return None

def is_valid_move(x, y):
    """Check if a move is within the puzzle boundaries."""
    return 0 <= x < SIZE and 0 <= y < SIZE

def perform_move(state, direction):
    """Perform a move if it's valid and return the new state."""
    x, y = find_blank_position(state)
    dx, dy = MOVES[direction]
    new_x, new_y = x + dx, y + dy
    if is_valid_move(new_x, new_y):
        new_state = deepcopy(state)
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state, direction  # Return the new state and the move made
    return None, None

def calculate_heuristic(state, goal):
    """Calculate the number of tiles in incorrect positions compared to the goal state."""
    incorrect = sum(1 for x in range(SIZE) for y in range(SIZE)
                    if state[x][y] != goal[x][y] and state[x][y] != '*')
    return incorrect

def solve_puzzle(state, goal, steps, path=[], depth=0):
    """Solve the puzzle using backtracking with heuristic guidance, explicitly showing backtracking."""
    if state == goal:
        steps.append(('Goal reached', state))
        return True
    if depth == DEPTH:
        return False  # Stop the recursion if the depth limit is reached
    
    possible_moves = []
    for direction in ['S', 'E', 'W', 'N']:
        new_state, _ = perform_move(state, direction)
        if new_state and new_state not in path:  # Avoid revisiting states
            heuristic = calculate_heuristic(new_state, goal)
            possible_moves.append((heuristic, new_state, direction))
    
    # Sort possible moves based on heuristic value
    possible_moves.sort(key=lambda x: x[0])

    for heuristic, new_state, direction in possible_moves:
        # Append current state and move to path before proceeding
        path.append((state, direction))
        steps.append((f'Move: {direction}, Heuristic: {heuristic}', new_state))

        if solve_puzzle(new_state, goal, steps, path, depth + 1):
            return True
        else:
            # Backtracking: remove the last move and state from path if it leads to a dead-end
            path.pop()
            steps.append((f'Backtracking from {direction}', state))

            # Check if the heuristic got worse, and if so, break the loop and backtrack
            if len(path) > 0 and len(steps) > 1 and heuristic >= calculate_heuristic(path[-1][0], goal):

                break

    return False


st.title('8 Puzzle Solver with Heuristic Backtracking')

# Inputs for initial and goal states
initial_state_input = st.text_input('Enter Initial State:', '6,2,3,8,5,*,4,1,7').split(',')
goal_state_input = st.text_input('Enter Goal State:', '8,6,2,4,5,3,*,1,7').split(',')

# Conversion to matrix form
initial_state_matrix = np.array(initial_state_input).reshape(SIZE, SIZE).tolist()
goal_state_matrix = np.array(goal_state_input).reshape(SIZE, SIZE).tolist()

# Button to start the solving process
if st.button('Solve Puzzle'):
    # Calculate heuristic for the initial state
    initial_heuristic = calculate_heuristic(initial_state_matrix, goal_state_matrix)
    steps = [(f"Initial state, Heuristic: {initial_heuristic}", initial_state_matrix)]

    # Start solving the puzzle
    if solve_puzzle(initial_state_matrix, goal_state_matrix, steps):
        for description, step in steps[:-1]:  # Display all but the final success message
            st.text(description)
            st.write(np.array(step))
        st.success(steps[-1][0])  # Display the final success or failure message
    else:
        st.error('No solution found or depth limit reached.')



# Copy and paste these, dont know how to get it show automitcally

# initial state 1
# 6,2,3,8,5,*,4,1,7

# goal state
# 8,6,2,4,5,3,*,1,7
        
# initial state 2
# 6,5,3,8,2,7,1,4,*