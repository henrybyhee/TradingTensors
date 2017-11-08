HIDDEN_LAYERS = [128, 64, 32]

#DQN PARAMETERS
GAMMA = 0.99
UPDATE_FREQUENCY = 500

#Regularization Params
L2_REG_LAMBDA = 0.9 #Higher reg, lower overfitting
KEEP_PROB = 0.1 # Dropout: How much neurons to keep?


#Epsilon-Greedy Algorithm
FINAL_P = 0.02
INITIAL_P = 1.0
