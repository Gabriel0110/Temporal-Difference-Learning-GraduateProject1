import random
import numpy as np
import matplotlib.pyplot as plt 

terminal_states = [0, 6] # 'A' or 'G'
starting_state = 3 # 'D'
n_sequences = 10
n_training_sets = 100
gamma = 1.0 # ADJUST

def takeAction(current_state):
    random.seed(0)
    state_dict = {'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3,
                  'E': 4,
                  'F': 5,
                  'G': 6,}
    if current_state == 3:
        next_state = random.choice([2, 4])
        reward = 0
    elif current_state == 2:
        next_state = random.choice([1, 3])
        reward = 0
    elif current_state == 4:
        next_state = random.choice([3, 5])
        reward = 0
    elif current_state == 1:
        next_state = random.choice([0, 2])
        reward = (-1 if next_state == 0 else 0)
    elif current_state == 5:
        next_state = random.choice([4, 6])
        reward = (1 if next_state == 6 else 0)
    return next_state, reward
 

def takeAction4(current_state):
    #random.seed(0)
    state_dict = {'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3,
                  'E': 4,
                  'F': 5,
                  'G': 6,}
    if current_state == 3:
        next_state = random.choice([2, 4])
        reward = 0
    elif current_state == 2:
        next_state = random.choice([1, 3])
        reward = 0
    elif current_state == 4:
        next_state = random.choice([3, 5])
        reward = 0
    elif current_state == 1:
        next_state = random.choice([0, 2])
        reward = (-1 if next_state == 0 else 0)
    elif current_state == 5:
        next_state = random.choice([4, 6])
        reward = (1 if next_state == 6 else 0)
    return next_state, reward

# SEQUENCE == EPISODE

# [A, B, C, D, E, F, G]

# [0, 1, 2, 3, 4, 5, 6]

################################################################################
#---------------------------------- FIGURE 3 ----------------------------------#

def runFigure3():
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha = 0.005 # ADJUST
    sequence = [starting_state]
    sequences = []
    sets_and_seqs = {}
    rmse_list = []
    rmse_for_each_run = [] # should be 100
    weights_for_sequences = []

    for lambd in lambdas:
        true_values = np.linspace(-1, 1, 7)[1:-1]
        V = np.zeros(7) # values/weights
        del rmse_for_each_run[:]
        for training_set in range(n_training_sets):
            delta = 0
            del weights_for_sequences[:]
            for seq in range(n_sequences):
                # Run algorithm for each sequence (10 per training set, 100 training sets)
                next_state = ''
                current_state = sequence[-1]  # initialize current state S
                Z = np.zeros(7) # initialize eligibility trace Z
                while next_state not in terminal_states:
                    next_state, reward = takeAction(current_state)  # choose and take action, observe reward and next state
                    Z = (Z * lambd * gamma)
                    Z[current_state] += 1
                    d_delta = reward + gamma * V[next_state] - V[current_state]
                    #V = V + alpha * delta * Z
                    current_state = next_state
                    sequence.append(next_state)
                delta += d_delta
                sequences.append(sequence) # add sequence to list of sequences
                del sequence[:] # reset sequence for next sequence run
                sequence.append(starting_state) # start new sequence with starting state again for next run
                weights_for_sequences.append(V[1:-1]) # only want non-terminal state weights
            V = V + alpha * delta * Z
            rmse_over_run = np.sqrt(np.mean((weights_for_sequences - true_values)**2))
            rmse_for_each_run.append(rmse_over_run)
            sets_and_seqs[training_set+1] = sequences
            del sequences[:]

        # Calculate error based on accumulated weights and true state values -- AVERAGE OVER ALL TRAINING SETS
        rmse = np.mean(np.array(rmse_for_each_run))
 
        # Store rmse in list of rmse's to plot onto graph with lambda values
        rmse_list.append(rmse)

    assert(len(rmse_list) == len(lambdas))
 
    #for i, lambd in enumerate(lambdas):
    #    plt.plot(lambdas[i], rmse_list[i], 'o', '-')
    plt.plot(lambdas, rmse_list, '-o')
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("Attempted Replication of Sutton Figure 3")
    plt.show()

#---------------------------------- FIGURE 3 ----------------------------------#
################################################################################




################################################################################
#---------------------------------- FIGURE 4 ----------------------------------#

def runFigure4():
    lambdas = [0.0, 0.3, 0.8, 1.0]
    alphas = np.linspace(0.0, 0.6, 11)
    sequence = [starting_state]
    sequences = []
    sets_and_seqs = {}
    rmse_list = []
    lambda_alphas_rmse_dict = {}
    rmse_for_each_run = [] # should be 100

    for lambd in lambdas:
        true_values = np.linspace(-1, 1, 7)[1:-1]
        sets_and_seqs.clear()
        del rmse_list[:]
        del rmse_for_each_run[:]
        for alpha in alphas:
            for training_set in range(n_training_sets):
                V = np.full((7), 0.5)
                weights_for_sequences = []
                for seq in range(n_sequences):
                    # Run algorithm for each sequence (10 per training set, 100 training sets)
                    next_state = ''
                    current_state = sequence[-1]  # initialize current state S
                    Z = np.zeros(7) # initialize eligibility trace Z
                    while next_state not in terminal_states:
                        next_state, reward = takeAction4(current_state)  # choose and take action, observe reward and next state
                        Z = (Z * lambd * gamma)
                        Z[current_state] += 1
                        delta = reward + gamma * V[next_state] - V[current_state]
                        current_state = next_state
                        sequence.append(next_state)
                        if len(sequence) >= 10:  # limit the sequence length
                            del sequence[:] # reset sequence for next sequence run
                            sequence.append(starting_state)
                            next_state = ''
                            current_state = sequence[-1]  # initialize current state S
                            Z = np.zeros(7)
                    V = V + alpha * delta * Z
                    sequences.append(sequence) # add sequence to list of sequences
                    del sequence[:] # reset sequence for next sequence run
                    sequence.append(starting_state) # start new sequence with starting state again for next run
                    weights_for_sequences.append(V[1:-1]) # only want non-terminal state weights
                rmse_over_run = np.sqrt(np.mean((weights_for_sequences - true_values)**2))
                rmse_for_each_run.append(rmse_over_run)
                sets_and_seqs[training_set+1] = sequences
                del sequences[:]
            del weights_for_sequences[:]

            # Calculate error based on accumulated weights and true state values -- AVERAGE OVER ALL TRAINING SETS
            rmse = np.mean(np.array(rmse_for_each_run))

            # Store rmse in list of rmse's to plot onto graph with each lambda value
            rmse_list.append(rmse) # RMSE VALUES FOR EACH ALPHA
        lambda_alphas_rmse_dict[lambd] = rmse_list[:] # APPENDING A LIST OF RMSE VALUES FOR DIFFERENT ALPHAS
        del weights_for_sequences[:]

    for lambd, errors in lambda_alphas_rmse_dict.items():
        plt.plot(alphas, errors, '-o', label=str(lambd))
    plt.xlabel("Alphas")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Attempted Replication of Sutton Figure 4")
    plt.show()

#---------------------------------- FIGURE 4 ----------------------------------#
################################################################################




################################################################################
#---------------------------------- FIGURE 5 ----------------------------------#

def runFigure5():
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha = 0.6 # best alpha from figure 4
    sequence = [starting_state]
    sequences = []
    sets_and_seqs = {}
    rmse_list = []
    rmse_for_each_run = [] # should be 100
    weights_for_sequences = []

    for lambd in lambdas:
        true_values = np.linspace(-1, 1, 7)[1:-1]
        V = np.zeros(7) # values/weights
        del rmse_for_each_run[:]
        for training_set in range(n_training_sets):
            delta = 0
            del weights_for_sequences[:]
            for seq in range(n_sequences):
                # Run algorithm for each sequence (10 per training set, 100 training sets)
                next_state = ''
                current_state = sequence[-1]  # initialize current state S
                Z = np.zeros(7) # initialize eligibility trace Z
                while next_state not in terminal_states:
                    next_state, reward = takeAction(current_state)  # choose and take action, observe reward and next state
                    Z = (Z * lambd * gamma)
                    Z[current_state] += 1
                    delta = reward + gamma * V[next_state] - V[current_state]
                    current_state = next_state
                    sequence.append(next_state)
                    if len(sequence) >= 10:  # limit the sequence length
                            del sequence[:] # reset sequence for next sequence run
                            sequence.append(starting_state)
                            next_state = ''
                            current_state = sequence[-1]  # initialize current state S
                            Z = np.zeros(7)
                V = V + alpha * delta * Z
                sequences.append(sequence) # add sequence to list of sequences
                del sequence[:] # reset sequence for next sequence run
                sequence.append(starting_state) # start new sequence with starting state again for next run
                weights_for_sequences.append(V[1:-1]) # only want non-terminal state weights
            rmse_over_run = np.sqrt(np.mean((weights_for_sequences - true_values)**2))
            rmse_for_each_run.append(rmse_over_run)
            sets_and_seqs[training_set+1] = sequences
            del sequences[:]

        # Calculate error based on accumulated weights and true state values -- AVERAGE OVER ALL TRAINING SETS
        rmse = np.mean(np.array(rmse_for_each_run))

        # Store rmse in list of rmse's to plot onto graph with lambda values
        rmse_list.append(rmse)
 
    #for i, lambd in enumerate(lambdas):
    #    plt.plot(lambdas[i], rmse_list[i], 'o', '-')
    plt.plot(lambdas, rmse_list, '-o')
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("Attempted Replication of Sutton Figure 5")
    plt.show()

#---------------------------------- FIGURE 5 ----------------------------------#
################################################################################

 
runFigure3()
runFigure4()
runFigure5()