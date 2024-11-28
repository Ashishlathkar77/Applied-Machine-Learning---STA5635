import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'map_24x32.csv'  
map_data = pd.read_csv(file_path, header=None).values

gamma = 0.9
epochs = 50

V = np.copy(map_data)
rows, cols = map_data.shape

actions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}

def display_value_function(V, epoch):
    plt.imshow(V, cmap='gray')
    plt.title(f'V(s) at Epoch {epoch}')
    plt.colorbar()
    plt.show()

# Part (a) Value Iteration
for epoch in range(1, epochs + 1):
    new_V = np.copy(V)
    
    for r in range(rows):
        for c in range(cols):
            if map_data[r, c] >= 0: 
                
                values = []
                for action, (dr, dc) in actions.items():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and map_data[nr, nc] >= 0:
                        reward = map_data[nr, nc]
                        values.append(reward + gamma * V[nr, nc])
                new_V[r, c] = max(values, default=0)
    
    V = new_V
    
    if epoch % 5 == 0 or epoch == 1:
        display_value_function(V, epoch)

# Part (b) Extract Final Policy
policy = np.full((rows, cols), '', dtype=str)

for r in range(rows):
    for c in range(cols):
        if map_data[r, c] >= 0:  
            best_action = None
            best_value = float('-inf')
            for action, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and map_data[nr, nc] >= 0:
                    reward = map_data[nr, nc]
                    value = reward + gamma * V[nr, nc]
                    if value > best_value:
                        best_value = value
                        best_action = action
            policy[r, c] = best_action if best_action else ' '

print("\nFinal Policy:")
for row in policy:
    print(' '.join(row))
