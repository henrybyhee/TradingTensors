import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.finance as mf
import numpy as np
import pandas as pd


class Q(object):
    def __init__(self, env, train_episodes=1000, learning_rate=0.2, gamma=0.9):
        self.train_episodes = train_episodes
        self.env = env
        self.lr = learning_rate
        self.y = gamma
        self.states_map = self.initialise_map(env.sim.state_labels)
        self.lookup_table = np.zeros([self.states_map.shape[0], env.action_space.n])

        

    def initialise_map(self, state_labels, sigma=0.5):
    #Return states_map
 
        #Define upper & lower bound in the observation space
        obs_high = np.ceil(self.env.observation_space.high)
        obs_low = np.floor(self.env.observation_space.low)

        #Find the number of evenly spaced intervals of sigma=0.5
        spaces = (obs_high - obs_low)/ sigma + 1
        
        if len(spaces)>1:
            table = []
            for i, space in enumerate(spaces):
                
                #For prices, create a list of numbers that is only separated by 0.1
                if 'Price' in state_labels[i]:
                    space = (obs_high[i] - obs_low[i])/0.1 + 1

                    price_list = np.linspace(obs_low[i], obs_high[i], space)

                    table.append(price_list)
                else:
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
        #Book-keeping tools
        self.journal_record = []
        self.reward_record = []
        self.avg_reward_record = []
        self.equity_curve_record = []

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
                
                #Perform update on self.lookup_table
                self.lookup_table[state, action] = (1. - self.lr) * self.lookup_table[state, action] + self.lr * (reward + self.y* max(self.lookup_table[next_state,:]))
                
                state = next_state

            if not train:
                start = self.env.sim.train_end_index + 1
                end = self.env.sim.count - 1
                print("End of Test Period from %s to %s, Total pips: %s, Average Reward is %s"%(self.env.sim.date_time[start], self.env.sim.date_time[end], self.env.portfolio.total_reward, self.env.portfolio.average_profit_per_trade))
            else:
                print("End of Episode %s, Total Reward is %s, Average Reward is %s"%(episode+1, self.env.portfolio.total_reward, self.env.portfolio.average_profit_per_trade))
                
            #Boring book-keeping after every episode
            self.journal_record.append(self.env.portfolio.journal)
            self.avg_reward_record.append(self.env.portfolio.average_profit_per_trade)
            self.reward_record.append(self.env.portfolio.total_reward)
            self.equity_curve_record.append(self.env.portfolio.equity_curve)

    def train(self):
        self.env._reset()
        self.run_episodes(self.train_episodes)

    def test(self, episodes=1):
        self.env._reset(train=False)
        self.run_episodes(episodes, False)

    def _overall_summary(self):
        #find episode with the highest run
        idx_hi_reward = np.argmax(self.reward_record)
        hi_reward = self.reward_record[idx_hi_reward]

        #Also find episode with the highest average run
        idx_avg_reward = np.argmax(self.avg_reward_record)
        avg_reward = self.avg_reward_record[idx_avg_reward]


        #Begin plotting function
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(20,10))

        ax1.set_title('Summary of Episodes')
        ax1.plot(self.reward_record, 'b-', label='Total Reward')
        ax1.annotate("Highest Reward: %s"%hi_reward, xy=(idx_hi_reward,hi_reward), xytext=(idx_hi_reward, hi_reward+100))
        ax1.legend(bbox_to_anchor=(0, 1), loc='upper left')

        ax2.plot(self.avg_reward_record, 'r-', label='Average Reward')
        ax2.set_xlabel('Episode')
        ax2.annotate("Best Performing: %s"%avg_reward, xy=(idx_avg_reward, avg_reward), xytext=(idx_avg_reward, avg_reward+2))
        ax2.legend(bbox_to_anchor=(0, 1), loc='upper left')
        
        plt.show()

        print("Episode with highest Reward: %s" %(idx_hi_reward+1))
        print("Episode with highest Average reward per trade: %s"%(idx_avg_reward+1))

    def episode_review(self, episode):
        
        index = episode - 1
        _journal = pd.DataFrame(self.journal_record[index])
        episode_reward = self.reward_record[index]
        episode_avg_reward = self.avg_reward_record[index]


        #Filter out Order Type
        buys = _journal.loc[_journal.Type == 'BUY', :]
        sells = _journal.loc[_journal.Type == 'SELL', :]

        print("SUMMARY OF EPISODE %s \n"% episode)
        
        print("Total Trades Taken: %s"%_journal.shape[0])
        print("%s Buys, %s Sells"%(buys.shape[0], sells.shape[0]))
        print("Total Reward: %s"%episode_reward)
        print("Reward: %s pips(Buy), %s pips(Sell) \n"%(buys['Profit'].sum(), sells['Profit'].sum()))

        print('Average Reward: %s'%episode_avg_reward)
        print("Win Ratio: %s %% \n"%(_journal.loc[_journal['Profit']>0, 'Profit'].count()/ _journal.shape[0]*100))


        #Data Preprocessing for candlestick charts
        data = self.env.sim._data

        ohlc = list(zip(mdates.date2num(data.index.to_pydatetime()), data.Open.tolist(), data.High.tolist(), data.Low.tolist(), data.Close.tolist()))
        

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(40,10))

        #First chart is Candlestick
        mf.candlestick_ohlc(ax1, ohlc, width= 0.02 , colorup='green', colordown='red')
        ax1.plot(buys['Entry Time'], buys['Entry Price']-0.001, 'b^', alpha=1.0)
        ax1.plot(sells['Entry Time'], sells['Entry Price']+0.001, 'rv', alpha=1.0)
        ax1.set_title('Candlestick Chart')

        #Second chart is Equity Curve
        ax2.plot(mdates.date2num(data.index.to_pydatetime()), self.equity_curve_record[index], '-')
        ax2.set_title('Equity Curve')
        plt.show()

        #Trade-by-Trade review
        print(_journal)




        