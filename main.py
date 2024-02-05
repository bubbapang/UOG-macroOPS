import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend; adjust if necessary based on your environment
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import permutations
from pyvirtualdisplay import Display

display = Display(visible=0, size=(800, 600))
display.start()

# Your matplotlib plotting code here

display.stop()


# Define the TSP distances array before generating cities coordinates
distances = [
    [0, 73, 29, 64, 90],
    [98, 0, 34, 16, 16],
    [99, 92, 0, 31, 98],
    [85, 54, 80, 0, 29],
    [53, 45, 73, 23, 0],
]

# Now, generate random city coordinates for visualization
# Assume some city coordinates for TSP visualization
cities = [
    (random.randint(0, 100), random.randint(0, 100)) for _ in range(len(distances))
]

def tsp_brute_force_with_path(distances):
    n = len(distances)
    min_path = float("inf")
    min_path_order = []
    for perm in permutations(range(n)):
        current_path = sum(distances[perm[i]][perm[i + 1]] for i in range(n - 1))
        path_length = current_path + distances[perm[-1]][perm[0]]
        if path_length < min_path:
            min_path = path_length
            min_path_order = perm
    return min_path, list(min_path_order)

def knapsack_with_items(values, weights, capacity):
    n = len(values)
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find the selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items

def plot_tsp_path(cities, path_order):
    plt.figure(figsize=(8, 6))
    # Plot the cities
    for (x, y) in cities:
        plt.plot(x, y, 'o')

    # Draw lines for the path
    for i in range(-1, len(path_order) - 1):
        x1, y1 = cities[path_order[i]]
        x2, y2 = cities[path_order[i + 1]]
        plt.plot([x1, x2], [y1, y2], 'r-')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Traveling Salesman Problem Solution')
    plt.savefig('tsp_solution.png')  # Saves the TSP plot
    plt.show()

def plot_knapsack_problem(values, weights, selected_indices):
    indexes = np.arange(len(values))
    selected_mask = np.zeros(len(values), dtype=bool)
    selected_mask[selected_indices] = True

    plt.figure(figsize=(10, 6))
    plt.bar(indexes, values, color='lightblue', label='Value')
    plt.bar(indexes[selected_mask], values[selected_mask], color='blue', label='Selected Value')
    plt.bar(indexes + 0.4, weights, width=0.4, color='lightgreen', alpha=0.5, label='Weight')
    plt.bar(indexes[selected_mask] + 0.4, np.array(weights)[selected_mask], width=0.4, color='green', alpha=0.5, label='Selected Weight')

    plt.xlabel('Item')
    plt.ylabel('Value / Weight')
    plt.title('Knapsack Problem Solution')
    plt.legend(loc='upper left')
    plt.xticks(indexes + 0.2, labels=[f"Item {i}" for i in indexes])
    plt.savefig('knapsack_solution.png')  # Saves the Knapsack plot, use in the respective function
    plt.show()

# Plotting solutions
# Solve the TSP to get the minimum distance and path
tsp_distance, tsp_path = tsp_brute_force_with_path(distances)

# Finally, call the visualization function with the cities and the path obtained from solving the TSP
plot_tsp_path(cities, tsp_path)
print(plt.get_backend())
