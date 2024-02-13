import streamlit as st
import numpy as np
from copy import deepcopy

# Configuration variables
SIZE = 3
DEPTH = 5

MOVES = {
    'S': (1, 0),  # South: Move down
    'E': (0, 1),  # East: Move right
    'W': (0, -1), # West: Move left
    'N': (-1, 0), # North: Move up
}

def find_blank_position(state):
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            if tile == '*':
                return (i, j)
    return None

def is_valid_move(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def perform_move(state, direction):
    x, y = find_blank_position(state)
    dx, dy = MOVES[direction]
    new_x, new_y = x + dx, y + dy
    if is_valid_move(new_x, new_y):
        new_state = deepcopy(state)
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state, direction  # Return the new state and the move made
    return None, None

def calculate_heuristic(state, goal):
    incorrect = sum(1 for x in range(SIZE) for y in range(SIZE)
                    if state[x][y] != goal[x][y] and state[x][y] != '*')
    return incorrect

def solve_puzzle(state, goal, steps, path=[], depth=0):
    if state == goal:
        steps.append(('Goal reached', state))
        return True
    if depth == DEPTH:
        return False  
    
    possible_moves = []
    for direction in ['S', 'E', 'W', 'N']:
        new_state, _ = perform_move(state, direction)
        if new_state and new_state not in path:  # Avoid revisiting states
            heuristic = calculate_heuristic(new_state, goal)
            possible_moves.append((heuristic, new_state, direction))
    
    # Sort possible moves based on heuristic value
    possible_moves.sort(key=lambda x: x[0])

    for heuristic, new_state, direction in possible_moves:
        path.append((state, direction))
        other_options = ", ".join([f"{d}, Heuristic: {h}" for h, _, d in possible_moves if d != direction])
        steps.append((f'Move: {direction}, Heuristic: {heuristic}, Other options: {other_options}', new_state))

        if solve_puzzle(new_state, goal, steps, path, depth + 1):
            return True
        else:
            path.pop()
            steps.append((f'Backtracking from {direction}', state))

    return False

st.title('8 Puzzle Solver with Heuristic Backtracking')

initial_state_input = st.text_input('Enter Initial State:', '6,2,3,8,5,*,4,1,7').split(',')
goal_state_input = st.text_input('Enter Goal State:', '8,6,2,4,5,3,*,1,7').split(',')

initial_state_matrix = np.array(initial_state_input).reshape(SIZE, SIZE).tolist()
goal_state_matrix = np.array(goal_state_input).reshape(SIZE, SIZE).tolist()

if st.button('Solve Puzzle'):
    initial_heuristic = calculate_heuristic(initial_state_matrix, goal_state_matrix)
    steps = [(f"Initial state, Heuristic: {initial_heuristic}", initial_state_matrix)]

    if solve_puzzle(initial_state_matrix, goal_state_matrix, steps):
        for description, step in steps[:-1]:
            st.text(description)
            st.write(np.array(step))
        st.success(steps[-1][0]) 
    else:
        st.error('No solution found within the depth limit.')
