import itertools
import numpy as np



class Q(object):
    def __init__(self, env, train_episodes=1000, learning_rate=0.2, gamma=0.9):
        self.train_episodes = train_episodes
        self.env = env
        self.lr = learning_rate
        self.y = gamma
        self.states_map = self.initialise_map()
        self.lookup_table = np.zeros([self.states_map.shape[0], env.action_space.n])


    def initialise_map(self, sigma=0.5):
    #Return states_map

        #Define upper & lower bound in the observation space
        obs_high = np.ceil(self.env.observation_space.high)
        obs_low = np.floor(self.env.observation_space.low)

        #Find the number of evenly spaced intervals of sigma=0.5
        spaces = (obs_high - obs_low)/ sigma + 1

        if len(spaces)>1:
            table = []
            for i, space in enumerate(spaces):
                table.append(np.linspace(obs_low[i], obs_high[i], space))

            states_map=[]
            for i in itertools.product(*table):
                states_map.append(i)
            states_map = np.array(states_map)

        else:
            states_map = np.linspace(obs_low, obs_high, spaces)
            
        return states_map

    #Approximate the observed state to a value on state_map
    def approx_state(self, observed):
        if observed.shape[0] == 1:
            return np.argmin(np.abs(self.states_map - observed))
        else:
            return np.argmin( np.abs(self.states_map - observed).sum(axis=1) )


    def run_episodes(self, episodes, train=True):
        if train:
            start = 0
            end = self.env.sim.train_end_index
            print("Training period  %s - %s"%(self.env.sim.date_time[start], self.env.sim.date_time[end]))
        for episode in range(episodes):

            self.env._reset(train)
            done = False
            start_index = self.env.sim.current_index

            #Initialise state
            start = self.env.sim.states[start_index-1]
            state = self.approx_state(start)
            
            while done is False:
                
                if self.env.portfolio.holding_trade:
                    #if we are still holding 
                    action = 2
                else:
                    #Otherwise, choose an action
                    choices = [0, 1, 2]
                    
                    #Pick the action with the highest value, plus exploration (Normally distributed values ~(0,1) )
                    exploration = np.random.randn(1, 3)[0]
                    action = np.argmax(self.lookup_table[state, choices] + exploration)

                #Step forward with the selected action
                obs, reward, done, info = self.env._step(action)
                
                #Estimate the next state based on observation generated
                
                next_state = self.approx_state(obs)
                
                
                   #Perform update on self.lookup_table only in train_mode
                if train:
                    self.lookup_table[state, action] = (1. - self.lr) * self.lookup_table[state, action] + self.lr * (reward + self.y* max(self.lookup_table[next_state,:]))
                state = next_state

            if not train:
                start = self.env.sim.train_end_index + 1
                end = self.env.sim.count - 1
                print("End of Test Period from %s to %s, Total Reward is %s"%(self.env.sim.date_time[start], self.env.sim.date_time[end], self.env.portfolio.total_reward))
            else:
                print("End of Episode %s, Total reward is %s"%(episode+1, self.env.portfolio.total_reward))

    def train(self):
        self.env._reset()
        self.run_episodes(self.train_episodes)

    def test(self, episodes=1):
        self.env._reset(train=False)
        self.run_episodes(episodes, False)

    def _generate_summary_stats(self):
        self.env._generate_summary_stats()