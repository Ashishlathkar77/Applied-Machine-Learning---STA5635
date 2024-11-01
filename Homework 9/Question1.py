import pandas as pd
import numpy as np


data = pd.read_csv('hmm_pb1.csv', header=None)
observed_sequence = data.iloc[0].values
observed_sequence


#Question 1: Part a)
states = [1, 2]  # 1: Fair, 2: Loaded
observations = [1, 2, 3, 4, 5, 6]  # Possible dice outcomes


# Transition probabilities
trans_probs = {
    1: {1: 0.95, 2: 0.05},  
    2: {1: 0.10, 2: 0.90}  
}


# Emission probabilities
emission_probs = {
    1: {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6},
    2: {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5}  
}


# Initial probabilities
initial_probs = {1: 0.5, 2: 0.5}


# Convert probabilities to log scale for numerical stability
log_initial_probs = {s: np.log(p) for s, p in initial_probs.items()}
log_trans_probs = {s: {s_next: np.log(p) for s_next, p in trans.items()} for s, trans in trans_probs.items()}
log_emission_probs = {s: {obs: np.log(p) for obs, p in emissions.items()} for s, emissions in emission_probs.items()}


# Step 3: Implement the Viterbi algorithm
def viterbi(observed_sequence, states, log_initial_probs, log_trans_probs, log_emission_probs):
    n = len(observed_sequence)
    T = len(states)


    viterbi_matrix = np.zeros((T, n))
    backpointer = np.zeros((T, n), dtype=int)


    for s in states:
        viterbi_matrix[s-1, 0] = log_initial_probs[s] + log_emission_probs[s][observed_sequence[0]]
        backpointer[s-1, 0] = 0


    for t in range(1, n):
        for s in states:
            max_tr_prob = viterbi_matrix[0, t-1] + log_trans_probs[1][s]
            prev_state_selected = 1
            for prev_s in states[1:]:
                tr_prob = viterbi_matrix[prev_s-1, t-1] + log_trans_probs[prev_s][s]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_state_selected = prev_s
            max_prob = max_tr_prob + log_emission_probs[s][observed_sequence[t]]
            viterbi_matrix[s-1, t] = max_prob
            backpointer[s-1, t] = prev_state_selected


    best_path_prob = np.max(viterbi_matrix[:, n-1])
    best_last_state = np.argmax(viterbi_matrix[:, n-1]) + 1


    best_path = np.zeros(n, dtype=int)
    best_path[-1] = best_last_state
    for t in range(n-2, -1, -1):
        best_path[t] = backpointer[best_path[t+1]-1, t+1]


    return best_path


# Step 4: Run the Viterbi algorithm
most_likely_sequence = viterbi(observed_sequence, states, log_initial_probs, log_trans_probs, log_emission_probs)


# Output the result
print("Most likely sequence of states (1=Fair, 2=Loaded):")
print(most_likely_sequence)


# Question 1: Part b)
import numpy as np
states = [1, 2]  # 1: Fair, 2: Loaded
observations = [1, 2, 3, 4, 5, 6]  # Possible dice outcomes


# Transition probabilities
trans_probs = {
    1: {1: 0.95, 2: 0.05},
    2: {1: 0.10, 2: 0.90}
}


# Emission probabilities
emission_probs = {
    1: {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6},
    2: {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.5}
}


# Initial probabilities
initial_probs = {1: 0.5, 2: 0.5}


# Step 1: Load observed sequence
T = len(observed_sequence)


# Forward algorithm with normalization
def forward_algorithm(observed_sequence, states, initial_probs, trans_probs, emission_probs):
    T = len(observed_sequence)
    forward_probs = np.zeros((len(states), T))
    u_t = np.zeros(T)


    for s in states:
        forward_probs[s-1, 0] = initial_probs[s] * emission_probs[s][observed_sequence[0]]
    u_t[0] = np.sum(forward_probs[:, 0])  
    forward_probs[:, 0] /= u_t[0]


    for t in range(1, T):
        for s in states:
            forward_probs[s-1, t] = sum(forward_probs[prev_s-1, t-1] * trans_probs[prev_s][s] for prev_s in states) \
                                    * emission_probs[s][observed_sequence[t]]
        u_t[t] = np.sum(forward_probs[:, t])  
        forward_probs[:, t] /= u_t[t]


    return forward_probs, u_t


# Backward algorithm with normalization
def backward_algorithm(observed_sequence, states, trans_probs, emission_probs, u_t):
    T = len(observed_sequence)
    backward_probs = np.zeros((len(states), T))


    backward_probs[:, T-1] = 1 / u_t[T-1]  


    for t in range(T-2, -1, -1):
        for s in states:
            backward_probs[s-1, t] = sum(backward_probs[next_s-1, t+1] * trans_probs[s][next_s] *
                                         emission_probs[next_s][observed_sequence[t+1]] for next_s in states)
        backward_probs[:, t] /= u_t[t]  


    return backward_probs


# Step 3: Run the forward and backward algorithms
alpha, u_t = forward_algorithm(observed_sequence, states, initial_probs, trans_probs, emission_probs)
beta = backward_algorithm(observed_sequence, states, trans_probs, emission_probs, u_t)


# Step 4: Report the required ratios
t_report = 139
alpha_ratio = alpha[0, t_report] / alpha[1, t_report]
beta_ratio = beta[0, t_report] / beta[1, t_report]


print("α^1_139 / α^2_139:", alpha_ratio)
print("β^1_139 / β^2_139:", beta_ratio)
