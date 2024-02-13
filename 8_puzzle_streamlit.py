import streamlit as st
import numpy as np
from copy import deepcopy

SIZE = 3
DEPTH = 5 # change if you want to go deeper, solve easier isntead of backtracking


MOVES = { # moves with directions
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'N': (-1, 0), 
}

def find_blank_position(state): # finds where the blank tile is
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            if tile == '*':
                return (i, j)
    return None

def is_valid_move(x, y): # can you move?
    return 0 <= x < SIZE and 0 <= y < SIZE

def perform_move(state, direction): # move the blank tile, check if good move
    x, y = find_blank_position(state)
    dx, dy = MOVES[direction]
    new_x, new_y = x + dx, y + dy
    if is_valid_move(new_x, new_y):
        new_state = deepcopy(state)
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state, direction
    return None, None

def calculate_heuristic(state, goal): # number of incorrect tiles
    incorrect = sum(1 for x in range(SIZE) for y in range(SIZE)
                    if state[x][y] != goal[x][y] and state[x][y] != '*')
    return incorrect

def solve_puzzle(state, goal, steps, path=[], depth=0): # recursively solve the puzzle and not pass the depth limit
    if state == goal:
        steps.append(('Goal reached', state, []))
        return True
    if depth == DEPTH:
        return False  
    
    possible_moves = []
    for direction in ['S', 'E', 'W', 'N']:
        new_state, _ = perform_move(state, direction)
        if new_state and new_state not in path:
            heuristic = calculate_heuristic(new_state, goal)
            possible_moves.append((heuristic, new_state, direction))
    
    # Sort
    possible_moves.sort(key=lambda x: x[0])

    for heuristic, new_state, direction in possible_moves:
        path.append((state, direction))
        other_options = [(h, s, d) for h, s, d in possible_moves if d != direction]
        steps.append((f'Move: {direction}, Heuristic: {heuristic}', new_state, other_options))

        if solve_puzzle(new_state, goal, steps, path, depth + 1):
            return True
        else:
            path.pop()
            steps.append((f'Backtracking from {direction}', state, []))

    return False

st.title('8 Puzzle Solver with Heuristic Backtracking') # srtreamlit stuff

initial_state_input = st.text_input('Enter Initial State:', '6,2,3,8,5,*,4,1,7').split(',')
goal_state_input = st.text_input('Enter Goal State:', '8,6,2,4,5,3,*,1,7').split(',')

initial_state_matrix = np.array(initial_state_input).reshape(SIZE, SIZE).tolist()
goal_state_matrix = np.array(goal_state_input).reshape(SIZE, SIZE).tolist()

if st.button('Solve Puzzle'): # button to solve the puzzle
    initial_heuristic = calculate_heuristic(initial_state_matrix, goal_state_matrix)
    steps = [(f"Initial state, Heuristic: {initial_heuristic}", initial_state_matrix, [])]

    if solve_puzzle(initial_state_matrix, goal_state_matrix, steps): # solve the puzzle
        for description, step, other_options in steps[:-1]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(description)
                st.write(np.array(step))
            with col2:
                if other_options:
                    st.write("Other options:")
                    for h, s, d in other_options:
                        st.text(f"Move: {d}, Heuristic: {h}")
                        st.write(np.array(s))
        st.success(steps[-1][0])
    else:
        st.error('No solution found within the depth limit. Displaying the first 10 steps attempted:') # for initial state 2
        # Display the first 5 steps or all steps if fewer
        for description, step, other_options in steps[:10]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(description)
                st.write(np.array(step))
            with col2:
                if other_options:
                    st.write("Other options:")
                    for h, s, d in other_options:
                        st.text(f"Move: {d}, Heuristic: {h}")
                        st.write(np.array(s))





# Copy and paste these, dont know how to get it show automitcally

# initial state 1
# 6,2,3,8,5,*,4,1,7

# goal state
# 8,6,2,4,5,3,*,1,7
        
# initial state 2
# 6,5,3,8,2,7,1,4,*