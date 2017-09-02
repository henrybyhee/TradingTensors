import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.finance as mf
import numpy as np
import pandas as pd

import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.finance as mf
import numpy as np
import pandas as pd
import os

import baselines.common.tf_util as U
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
import tensorflow as tf
from .deepq import DQN, mini_batch_training, q_act, update_target_network




class LearningAgent(object):
    '''Parent Class of all Learning Agents
    
    '''

    def __init__(self, identity, train_episodes, env):
        self.env = env
        self.train_episodes = train_episodes
        self.Oanda = False
        self.identity = identity

        
    def reset_bookkeeping_tools(self):
        #Book-keeping tools
        self.journal_record = []
        self.reward_record = []
        self.avg_reward_record = []
        self.equity_curve_record = []
        
    def _overall_summary(self):
        '''Common Function to display learning curves of the  policy'''

        #find episode with the highest run
        idx_hi_reward = np.argmax(self.reward_record)
        hi_reward = self.reward_record[idx_hi_reward]

        #Also find episode with the highest average run
        idx_avg_reward = np.argmax(self.avg_reward_record)
        avg_reward = self.avg_reward_record[idx_avg_reward]

        #Begin plotting function
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(20,10))

        ax1.set_title('Summary of %s Episodes'%(self.identity))
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

        #Compute the episode with the most optimal policy
        print("Episode with the most optimal policy (average reward * total reward): %s"%(self.best_index))
    
    def episode_review(self, episode):
        
        index = episode - 1
        _journal = pd.DataFrame(self.journal_record[index])
        episode_reward = self.reward_record[index]
        episode_avg_reward = self.avg_reward_record[index]


        #Filter out Order Type
        buys = _journal.loc[_journal['Type'] == 'BUY', :]
        sells = _journal.loc[_journal['Type'] == 'SELL', :]

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
    
    def trade_with_Oanda(self, units=1, train=True):
        
        self.Oanda = True
        self.env._reset(train, self.Oanda)    
        self.env.sim.trade_with_Oanda()
        
        data4order = {
            "order": {
                    "timeInForce": "FOK",
                    "units": units,
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
            }
        }
        
        self.env.portfolio.set_client(self.env.sim.client, self.env.sim.accountID, data4order, self.env.sim.instrument)
        
        self.run_episodes(1, train=train)
    

class Q(LearningAgent):
    '''Q_table implementation'''
    def __init__(self, env, train_episodes=1000, learning_rate=0.2, gamma=0.9, discret_sigma=0.5):
        
        super().__init__('Q_Table', train_episodes, env)
        
        self.lr = learning_rate
        self.y = gamma
        self.states_map = self.initialise_map(env.sim.state_labels, sigma=discret_sigma)
        self.lookup_table = np.zeros([self.states_map.shape[0], env.action_space.n])
        self.best_table = self.lookup_table

    def initialise_map(self, state_labels, sigma=0.5):
    #Return states_map
 
        #Define upper & lower bound in the observation space
        obs_high = np.ceil(self.env.observation_space.high)
        obs_low = np.floor(self.env.observation_space.low)

        #Find the number of evenly spaced intervals of sigma=0.5
        _spaces = (obs_high - obs_low)/ sigma + 1
        
        if len(_spaces)>1:
            table = []
            for i, space in enumerate(_spaces):
                
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
            states_map = np.linspace(obs_low, obs_high, _spaces)
            
        return states_map

    #Approximate the observed state to a value on state_map
    def approx_state(self, observed):
        if observed.shape[0] == 1:
            return np.argmin(np.abs(self.states_map - observed))
        else:
            return np.argmin( np.abs(self.states_map - observed).sum(axis=1) )


    def run_episodes(self, episodes, optimal_table=None, train=True, policy_measure='optimal', EPSILON=0.1):

        self.reset_bookkeeping_tools()

        if train:
            start = 0
            end = self.env.sim.train_end_index
            print("Training period  %s - %s"%(self.env.sim.date_time[start], self.env.sim.date_time[end]))

        max_reward = 0
        
        if not optimal_table is None:
            #Initialise the lookup table with the best policy before running episodes
            self.lookup_table = optimal_table

        
        for episode in range(episodes):

            self.env._reset(train, self.Oanda)
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
                    
                    
                    if np.abs(np.random.randn()) < EPSILON:
                        #Exploration
                        action = np.random.randint(0,2)
                        
                    else:
                        #Exploitation
                        action = np.argmax(self.lookup_table[state, choices])
                        
    
                #Step forward with the selected action
                obs, reward, done, info = self.env._step(action)
                
                #Estimate the next state based on observation generated
                next_state = self.approx_state(obs)
                
                #Perform update on self.lookup_table
                self.lookup_table[state, action] = self.lookup_table[state, action]  + self.lr * (reward + self.y* max(self.lookup_table[next_state,:]) - self.lookup_table[state, action])
                
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

            #Scoring the Total Reward and Average reward collected by the agent
            if policy_measure == 'average':
                score = self.avg_reward_record[-1]
            elif policy_measure == 'highest':
                score = self.reward_record[-1]
            else:
                score = np.abs(self.avg_reward_record[-1])*self.reward_record[-1]

            #Save the best performing learning policy in the form of table
            if score > max_reward:
                self.best_table = self.lookup_table
                self.best_index = episode
                max_reward = score

            #Decrease epsilon to cutdown on exploration once it has crossed 100 episodes
            if episode > 100:
                EPSILON = EPSILON / episode


    def train(self, policy_measure='optimal'):
        assert policy_measure in ['optimal', 'highest','average'], "Policy_measure can only be either 'optimal', 'highest' or 'average'!"
        
        self.Oanda = False
        self.env._reset(train=True, Oanda=self.Oanda)
        self.run_episodes(self.train_episodes, policy_measure=policy_measure)
        
    def test(self, episodes=1, policy_measure='optimal'):
        assert policy_measure in ['optimal', 'highest','average'], "Policy_measure can only be either 'optimal', 'highest' or 'average'!"

        self.Oanda = False
        self.env._reset(train=False, Oanda=self.Oanda)
        self.run_episodes(episodes, False, policy_measure=policy_measure)

    def save_this_policy(self):
        np.save('best_q_policy.npy', [self.best_index, self.reward_record[self.best_index], self.avg_reward_record[self.best_index],self.best_table])

    def compare_policies(self):
        if os.path.exists('best_q_policy.npy') is False:
            print("No past record of policies saved")
            return 
        else:
            lastIndex, lastReward, lastAvg, lastTable = np.load('optimal_q_policy.npy')
            print("Saved Policy: %s (Total Reward), %s (Average Reward)"%(lastReward, lastAvg))
            print("Current Best Policy: %s (Total Reward), %s (Average Reward)"%(self.reward_record[self.best_index], self.avg_reward_record[self.best_index]))










class Q_Network(LearningAgent):
    def __init__(self, env, hidden_layers=[128, 64, 32] , train_episodes=1500, update_frequency=500):

        super().__init__('Q_network', train_episodes, env)

        self.path = './saved_tensor_models/dqn.ckpt'

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        self.neurons = hidden_layers
        self.frequency = update_frequency
        
        self.online_network = DQN(env, hidden_layers, 'online')
        self.target_network = DQN(env, hidden_layers, 'target')
        self.Oanda = False

    def train_model(self, batch_size=32, policy_measure='optimal', convergence_threshold=500, episodes_to_explore=100):
        
        self.env._reset(train=True, Oanda=self.Oanda)
        steps_per_episode = self.env.sim._end - self.env.sim.current_index
        total_steps = steps_per_episode * self.train_episodes
        

        exploration = LinearSchedule(steps_per_episode * episodes_to_explore, final_p=0.02, initial_p=1.0)
        replaybuffer = ReplayBuffer(total_steps*1.2)

        #Use of parallelism 
        config_proto=tf.ConfigProto(
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
            )
        with tf.Session(config=config_proto) as sess:
            
            sess.run(tf.global_variables_initializer())
            self.online_network, self.target_network = update_target_network(sess,self.online_network,self.target_network)
            
            
            saver = tf.train.Saver()
            t = 0

            self.reset_bookkeeping_tools()  
            max_reward = 0 
            self.best_index = 0

            for epi in range(self.train_episodes):
                
                self.env._reset(train=True, Oanda=self.Oanda)
        
                state = self.env.sim.states[0]

                done = False
                solved = False
                action_dict = {0:0, 1:0, 2:0}


                print("Training Period: %s - %s"%(self.env.sim.date_time[0], self.env.sim.date_time[self.env.sim.train_end_index]))
                
                while not done:
                    
                    #Predict action given this observation, with random chance of episilon (Exploration)
                    action = q_act(state, self.online_network, exploration.value(t), self.env, sess)

                    #if we are still holding a trade, as specified by the trade_period
                    if self.env.portfolio.holding_trade:
                        action = 2

                    action_dict[action] += 1

                    #Obtain next state and  reward with action
                    new_state, reward, done, _ = self.env._step(action)
                    
                    #Store this transition in memory
                    replaybuffer.add(state, action, reward, new_state, float(done))

                    state = new_state
                    t += 1

                    if t > 500:
                        #Optimize Online network with SGD
                        self.online_network, self.target_network = mini_batch_training(sess, self.env, self.online_network, self.target_network, replaybuffer, BATCH_SIZE=batch_size)
                        
                    if t % 500 == 0:
                        #Periodically update target network with online network
                        self.online_network, self.target_network = update_target_network(sess, self.online_network, self.target_network)

                    if done:

                        #Boring book-keeping after every episode
                        self.journal_record.append(self.env.portfolio.journal)
                        self.avg_reward_record.append(self.env.portfolio.average_profit_per_trade)
                        self.reward_record.append(self.env.portfolio.total_reward)
                        self.equity_curve_record.append(self.env.portfolio.equity_curve)
                        

                        print("End of Episode %s, Total Reward is %s, Average Reward is %.3f"%(epi+1, self.env.portfolio.total_reward, self.env.portfolio.average_profit_per_trade))
                        print("Percentage of time spent on exploring (Random Action): %s %%"%(int(100 * exploration.value(t))))
                        print(action_dict)
                        assert policy_measure in ['average', 'highest', 'optimal'], "policy measure can only be 'average', 'highest', or 'optimal'"
                        
                        if policy_measure == 'average':
                            score = self.avg_reward_record[-1]
                        elif policy_measure == 'highest':
                            score = self.reward_record[-1]
                        elif policy_measure == 'optimal':
                            score = np.abs(self.avg_reward_record[-1])*self.reward_record[-1]
                        

                        if score > max_reward and exploration.value(t) < 0.1:
                            #Save model if there is a new max_reward, with little exploration
                            print("New Maximum Score found! Saving this model. Score: %s"%score)
                            saver.save(sess, self.path)
                            max_reward = score
                            self.best_index = epi

                        print()
                        #Check for Convergence
                        if np.mean(self.reward_record[-51:-1]) > convergence_threshold and exploration.value(t) < 0.04:
                            solved = True
                            

                if solved:
                    print("Converged!")
                    print()
                    break
    
    def test_model(self):
        '''No Optimization is performed during Testing phase'''        

        if not os.path.exists(self.path +'.meta'):
            print("No .meta files are found for this model %s, please train the network."%(self.path))
            return

        with tf.Session() as sess:
            #Use previously run model
            saver = tf.train.Saver()
            saver.restore(sess, self.path)

            self.reset_bookkeeping_tools()
            self.env._reset(train=False, Oanda=self.Oanda)


            state = self.env.sim.states[self.env.sim.current_index - 1]
            done = False

            while not done:
                #Feedforward thru Network to obtain action
                action = sess.run(self.online_network.Q_action, feed_dict={
                    self.online_network._inputs: np.reshape(state, [1, self.env.observation_space.shape[0]])
                    })[0]

                #Step forward in the environment
                next_state, reward, done, _ = self.env._step(action)

                state = next_state


            #Once Complete
            #Boring book-keeping
            self.journal_record.append(self.env.portfolio.journal)
            self.avg_reward_record.append(self.env.portfolio.average_profit_per_trade)
            self.reward_record.append(self.env.portfolio.total_reward)
            self.equity_curve_record.append(self.env.portfolio.equity_curve)

            print("End of Testing, Total Reward is %s, Average Reward is %s"%(self.env.portfolio.total_reward, self.env.portfolio.average_profit_per_trade))


    def trade_with_Oanda(self, units=1):
        
        self.Oanda = True  
        data4order = {
            "order": {
                    "timeInForce": "FOK",
                    "units": units,
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
            }
        }
        
        self.env.portfolio.set_client(self.env.sim.client, self.env.sim.accountID, data4order, self.env.sim.instrument)
        
        self.env.sim.trade_with_Oanda()
        
        self.test_model()
        



        
