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

def solve_puzzle(state, goal, steps, path=[], depth=0, max_steps=10):
    """Solve the puzzle using backtracking, showing steps accurately."""
    if depth == 0 and steps:
        # Reset steps list on the first call, except for the initial state already added
        steps.clear()
        steps.append(("Initial state", state))

    if state == goal:
        steps.append(('Goal reached', state))
        return True

    if depth >= DEPTH or len(steps) >= max_steps:
        # Exit if depth exceeds limit or enough steps are taken
        return False

    for direction in ['S', 'E', 'W', 'N']:
        if len(steps) >= max_steps:
            # Ensure not to exceed the step limit
            break

        new_state, move_made = perform_move(state, direction)
        if new_state and new_state not in path:  # Avoid cycles by not revisiting states
            heuristic = calculate_heuristic(new_state, goal)
            if len(steps) < max_steps:  # Check to ensure within step limit before adding
                steps.append((f"Move: {direction} (Heuristic: {heuristic})", new_state))
            
            if solve_puzzle(new_state, goal, steps, path + [state], depth + 1, max_steps):
                return True
            elif len(steps) < max_steps:
                # Note backtracking only if within steps limit and if the move didn't succeed
                steps.append((f'Backtracking from {direction}', state))

    # If no solution found within the depth or steps limit, ensure to report stopping reason
    if depth == 0 and len(steps) == 1:  # Only the initial state was added, and no solution found
        steps.append(('Solution not found within step limit', state))
    return False



st.title('8 Puzzle Solver with Heuristic Backtracking')

initial_state_input = st.text_input('Enter Initial State:', '6,5,3,8,2,7,1,4,*').split(',')
goal_state_input = st.text_input('Enter Goal State:', '8,6,2,4,5,3,*,1,7').split(',')

initial_state_matrix = np.array(initial_state_input).reshape(SIZE, SIZE).tolist()
goal_state_matrix = np.array(goal_state_input).reshape(SIZE, SIZE).tolist()

if st.button('Solve Puzzle'):
    steps = [("Initial state", initial_state_matrix)]  # Start with the initial state description

    solve_puzzle(initial_state_matrix, goal_state_matrix, steps)

    # Ensure to display exactly up to the first 10 steps
    for description, step in steps[:50]:
        st.text(description)
        st.write(np.array(step))

    if len(steps) > 50:
        st.warning("Displayed the first 10 steps. The solution process exceeded this limit.")
    elif steps[-1][0] == 'Goal reached':
        st.success("Puzzle solved within the first 10 steps!")
    else:
        st.error("Puzzle not solved within the first 10 steps.")


# Copy and paste these, dont know how to get it show automitcally

# initial state 1
# 6,2,3,8,5,*,4,1,7

# goal state
# 8,6,2,4,5,3,*,1,7
        
# initial state 2
# 6,5,3,8,2,7,1,4,*