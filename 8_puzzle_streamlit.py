import streamlit as st
import numpy as np
from copy import deepcopy

#size of puzzle
PUZZLE_SIZE = 3

#order of the moves
MOVES = {
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'N': (-1, 0),
}

# empty space, the star thing
def find_blank_position(state):
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            if state[i][j] == '*':
                return (i, j)
    return None

# is it a valid move
def is_valid_move(x, y):
    return 0 <= x < PUZZLE_SIZE and 0 <= y < PUZZLE_SIZE

# Perform a move
def perform_move(state, direction):
    x, y = find_blank_position(state)
    dx, dy = MOVES[direction]
    new_x, new_y = x + dx, y + dy
    if is_valid_move(new_x, new_y):
        new_state = deepcopy(state)
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state
    return None

# Backtracking algorithm to solve the puzzle
def solve_puzzle(state, goal, steps):
    if state == goal:
        return True
    for direction in ['S', 'E', 'W', 'N']:  # Order of moves: S > E > W > N
        new_state = perform_move(state, direction)
        if new_state and new_state not in steps:
            steps.append(new_state)
            if solve_puzzle(new_state, goal, steps):
                return True
            steps.pop()
    return False

# Streamlit app
st.title('8 Puzzle Solver with Backtracking')

def display_state(state_matrix, title):
    st.subheader(title)
    for row in state_matrix:
        # Replace '*' with ' ' for display
        display_row = [x if x != '*' else ' ' for x in row]
        st.text(' '.join(display_row))

# Input for initial state
initial_state_input = st.text_input('Enter Initial State (e.g., 1,2,3,4,5,6,7,8,* for the solved state):', '1,2,3,4,5,6,7,8,*')
initial_state = initial_state_input.split(',')
initial_state_matrix = np.array(initial_state).reshape(PUZZLE_SIZE, PUZZLE_SIZE)

# Input for goal state
goal_state_input = st.text_input('Enter Goal State (e.g., 1,2,3,4,5,6,7,8,* for the solved state):', '1,2,3,4,5,6,7,8,*')
goal_state = goal_state_input.split(',')
goal_state_matrix = np.array(goal_state).reshape(PUZZLE_SIZE, PUZZLE_SIZE)

display_state(initial_state_matrix, 'Initial State:')
display_state(goal_state_matrix, 'Goal State:')

if st.button('Solve Puzzle'):
    steps = [initial_state_matrix.tolist()]
    if solve_puzzle(initial_state_matrix.tolist(), goal_state_matrix.tolist(), steps):
        st.success('Solution found!')
        for step in steps:
            # Replace '*' with ' ' for display
            display_step = [[x if x != '*' else ' ' for x in row] for row in step]
            st.write(np.array(display_step))
    else:
        st.error('No solution found.')
