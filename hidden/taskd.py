import numpy as np

def fFrog(x, y):
    return x * np.cos(np.sqrt(abs(x + y + 1))) * np.sin(np.sqrt(abs(y - x + 1))) + \
           (1 + y) * np.sin(np.sqrt(abs(x + y + 1))) * np.cos(np.sqrt(abs(y - x + 1)))

def randomized_hill_climbing(sp, p, z, seed):
    np.random.seed(seed)
    
    # Use sp as the starting point
    current_solution = np.array(sp)
    current_value = fFrog(current_solution[0], current_solution[1])
    function_calls = 1  # Initial call to fFrog
    
    while True:
        neighbors = []
        
        for _ in range(p):
            v = np.random.uniform(-z, z, 2)
            neighbor = current_solution + v
            neighbors.append((neighbor, fFrog(neighbor[0], neighbor[1])))
            function_calls += 1  # Increment function call count for each neighbor
        
        best_neighbor = min(neighbors, key=lambda x: x[1])
        
        if best_neighbor[1] < current_value:
            current_solution = best_neighbor[0]
            current_value = best_neighbor[1]
        else:
            break
    
    return current_solution, current_value, function_calls

def RHCR2(sp, z, p, seed):
    total_function_calls = 0
    
    # First run
    sol1, f_sol1, calls1 = randomized_hill_climbing(sp, p, z, seed)
    total_function_calls += calls1
    
    # Second run with z/20
    sol2, f_sol2, calls2 = randomized_hill_climbing(sol1, p, z/20, seed)
    total_function_calls += calls2
    
    # Third run with z/400
    sol3, f_sol3, calls3 = randomized_hill_climbing(sol2, p, z/400, seed)
    total_function_calls += calls3
    
    return [(sol1, f_sol1), (sol2, f_sol2), (sol3, f_sol3)], total_function_calls

# Initialize best result tracking variables
best_result = None
best_value = float('inf')
best_params = None
best_function_calls = 0

# Given parameters
starting_points = [(-300, -400), (0, 0), (-222, -222), (-510, 400)]
ps = [120, 400]
zs = [9, 50]
seeds = [42, 43]

# Loop through all combinations and find the best result
for sp in starting_points:
    for p in ps:
        for z in zs:
            for seed in seeds:
                results, function_calls = RHCR2(sp, z, p, seed)
                sol3, f_sol3 = results[-1]  # Correctly unpack the final result
                
                if f_sol3 < best_value:
                    best_result = results
                    best_value = f_sol3
                    best_params = (sp, z, p, seed)
                    best_function_calls = function_calls

print(f"Best result: {best_result} with parameters sp={best_params[0]}, z={best_params[1]}, p={best_params[2]}, seed={best_params[3]}")
print(f"Function was called {best_function_calls} times in total.")
