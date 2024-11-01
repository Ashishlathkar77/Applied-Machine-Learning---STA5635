# Question 2
import numpy as np
import pandas as pd


# Define states and observations
states = [1, 2]  # 1: Fair, 2: Loaded
observations = [1, 2, 3, 4, 5, 6]  # Possible dice outcomes


data = pd.read_csv('/content/hmm_pb2.csv', header=None)
observed_sequence = data.iloc[0].values  
T = len(observed_sequence)


pi = np.array([0.5, 0.5])


a = np.array([[0.95, 0.05],  
              [0.10, 0.90]])


b = np.zeros((len(states), len(observations)))


b[0] = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])


b[1] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])


b = b / b.sum(axis=1, keepdims=True)


# Function for Baum-Welch algorithm
def baum_welch(observed_sequence, states, pi, a, b, max_iter=100):
    T = len(observed_sequence)
    num_states = len(states)
    num_observations = len(observations)


    for iteration in range(max_iter):
        # Step 1: Run forward algorithm
        alpha, u_t = forward_algorithm(observed_sequence, states, {1: pi[0], 2: pi[1]},
                                       {1: {1: a[0, 0], 2: a[0, 1]}, 2: {1: a[1, 0], 2: a[1, 1]}},
                                       {1: {1: b[0, 0], 2: b[0, 1], 3: b[0, 2], 4: b[0, 3], 5: b[0, 4], 6: b[0, 5]},
                                        2: {1: b[1, 0], 2: b[1, 1], 3: b[1, 2], 4: b[1, 3], 5: b[1, 4], 6: b[1, 5]}})


        # Step 2: Run backward algorithm
        beta = backward_algorithm(observed_sequence, states,
                                  {1: {1: a[0, 0], 2: a[0, 1]}, 2: {1: a[1, 0], 2: a[1, 1]}},
                                  {1: {1: b[0, 0], 2: b[0, 1], 3: b[0, 2], 4: b[0, 3], 5: b[0, 4], 6: b[0, 5]},
                                   2: {1: b[1, 0], 2: b[1, 1], 3: b[1, 2], 4: b[1, 3], 5: b[1, 4], 6: b[1, 5]}}, u_t)


        # Step 3: Calculate expected values
        expected_counts = np.zeros((num_states, T))
        expected_transitions = np.zeros((num_states, num_states))
        expected_emissions = np.zeros((num_states, num_observations))


        for t in range(T):
           
            for i in states:
                expected_counts[i-1, t] = alpha[i-1, t] * beta[i-1, t]


        for t in range(T-1):
            for i in states:
                for j in states:
                    expected_transitions[i-1, j-1] += (alpha[i-1, t] * a[i-1, j-1] *
                                                       b[j-1][observed_sequence[t+1]-1] * beta[j-1, t+1])
            for i in states:
                expected_emissions[i-1, observed_sequence[t]-1] += expected_counts[i-1, t]


        # Step 4: Update parameters
        pi = expected_counts[:, 0] / np.sum(expected_counts[:, 0])  # Initial probabilities
        for i in states:
            for j in states:
                expected_transitions[i-1, j-1] /= np.sum(expected_counts[i-1, :-1])  # Transition probabilities


        for i in states:
            expected_emissions[i-1] /= np.sum(expected_counts[i-1])  # Emission probabilities


    return pi, a, b


# Run Baum-Welch algorithm
pi, a, b = baum_welch(observed_sequence, states, pi, a, b)


# Report the obtained values
print("Estimated initial probabilities (π):", pi)
print("Estimated transition probabilities (a):\n", a)
print("Estimated emission probabilities (b):\n", b)
