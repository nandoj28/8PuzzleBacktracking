import csv
import numpy as np

def fFrog(x, y):
    return x * np.cos(np.sqrt(abs(x + y + 1))) * np.sin(np.sqrt(abs(y - x + 1))) + \
           (1 + y) * np.sin(np.sqrt(abs(x + y + 1))) * np.cos(np.sqrt(abs(y - x + 1)))

def randomized_hill_climbing(sp, p, z, seed):
    np.random.seed(seed)
    
    current_solution = np.array(sp)
    current_value = fFrog(current_solution[0], current_solution[1])
    function_calls = 1 
    
    while True:
        neighbors = []
        
        for _ in range(p):
            v = np.random.uniform(-z, z, 2)
            neighbor = current_solution + v
            neighbors.append((neighbor, fFrog(neighbor[0], neighbor[1])))
            function_calls += 1 
        
        best_neighbor = min(neighbors, key=lambda x: x[1])
        
        if best_neighbor[1] < current_value:
            current_solution = best_neighbor[0]
            current_value = best_neighbor[1]
        else:
            break
    
    return current_solution, current_value, function_calls

def RHCR2(sp, z, p, seed):
    sol1, f_sol1, calls1 = randomized_hill_climbing(sp, p, z, seed)
    sol2, f_sol2, calls2 = randomized_hill_climbing(sol1, p, z/20, seed)
    sol3, f_sol3, calls3 = randomized_hill_climbing(sol2, p, z/400, seed)
    
    total_function_calls = calls1 + calls2 + calls3
    
    return [(sol1, f_sol1, calls1), (sol2, f_sol2, calls2), (sol3, f_sol3, calls3)], total_function_calls

starting_points = [(-300, -400), (0, 0), (-222, -222), (-510, 400)] # adjust params if you want to try one specific value for each of the parms
ps = [120, 400]                                                     # like for the 33rd run
zs = [9, 50]
seeds = [43] # choose your seed, in this case. 42 or 43

all_results = []

for sp in starting_points:
    for p in ps:
        for z in zs:
            for seed in seeds:
                results, total_calls = RHCR2(sp, z, p, seed)
                all_results.append({
                    "params": {"sp": sp, "z": z, "p": p, "seed": seed},
                    "results": results,
                    "total_calls": total_calls
                })

csv_file_path = './optimization_results1.csv'  # Specify your CSV file name and path here, rename for different seeds

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SP_x', 'SP_y', 'Z', 'P', 'Seed', 'Solution_x', 'Solution_y', 'F(Solution)', 'Function Calls', 'Run', 'Total Function Calls'])

    for result in all_results:
        params = result["params"]
        total_calls = result["total_calls"]
        for i, (sol, f_sol, calls) in enumerate(result["results"], 1):
            writer.writerow([
                params['sp'][0], params['sp'][1], params['z'], params['p'], params['seed'],
                sol[0], sol[1], f_sol, calls, i, total_calls
            ])

print(f"CSV file has been created: {csv_file_path}")