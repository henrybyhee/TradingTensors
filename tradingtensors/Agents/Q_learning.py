import os
import time
from datetime import datetime
from queue import LifoQueue
from threading import Thread

import numpy as np
import pandas as pd
import tensorflow as tf

from ..settings.DQNsettings import (FINAL_P, GAMMA, INITIAL_P,
                                    UPDATE_FREQUENCY)
from .BaseQ import (DQN, LinearDecay, ReplayBuffer, choose_action,
                    mini_batch_training, update_target_network)
from .visual_utils import ohlcPlot, rewardPlot


class LearningAgent(object):
    '''Parent Class of Learning Agents
    '''

    def __init__(self, env, IDENTITY):
        self.env = env
        self.IDENTITY = IDENTITY

    def reset_bookkeeping_tools(self):
        #Book-keeping tools
        self.journal_record = []
        self.reward_record = []
        self.avg_reward_record = []
        self.equity_curve_record = []


class DQNAgent(LearningAgent):

    def __init__(
        self, env, PARENT_PATH, hidden_layers=[128, 64, 32]
        ):

        #Base Learning Agent initialization
        super().__init__(
            env, 
            'Deep Q Network',
        )


        self.tensor_dir_template =None
        self.PARENT_PATH = PARENT_PATH

        self.neurons = hidden_layers
        self.UPDATE_FREQUENCY = UPDATE_FREQUENCY

        self.online_net = DQN(env, hidden_layers, 'online')
        self.target_net = DQN(env, hidden_layers, 'target')

        self.best_models = []


    def clearTensorFolder(self):
        '''Remove all saved tensor model in the folder'''
        for dir_ in os.listdir(self.PARENT_PATH):
            os.remove(os.path.join(self.PARENT_PATH, dir_))

    
    def train(
        self, 
        policy_measure='optimal',
        BATCH_SIZE = 32,
        CONVERGENCE_THRESHOLD = 2000,
        EPISODES_TO_EXPLORE = 30,
        TRAIN_EPISODES = 200
        ):
        '''
        Run the full training cycle
        '''

        assert policy_measure in ['average', 'highest', 'optimal'], \
        "policy measure can only be 'average', 'highest', or 'optimal'"

        #Define saved model directory
        TIMESTAMP = datetime.fromtimestamp(
            time.time()
            ).strftime('%H%M')

        self.tensor_dir_template = os.path.join(
            self.PARENT_PATH, 
            TIMESTAMP+'_Episode%s.ckpt')

        #Clear all previous tensor models
        self.clearTensorFolder()


        #Estimate total steps
        STEPS_PER_EPISODE = \
            self.env.sim.train_end_idx - self.env.sim.train_start_idx - 2

        TOTAL_STEPS = STEPS_PER_EPISODE * TRAIN_EPISODES


        #Create a Transition memory storage
        replaybuffer = ReplayBuffer(TOTAL_STEPS*1.2)

        #Use of parallelism 
        config_proto=tf.ConfigProto(
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
            )

        #Keep track of top 10 models
        current_top_10s = []

        with tf.Session(config=config_proto) as sess:

            #Initialize all weights and biases in NNs
            sess.run(tf.global_variables_initializer())

            #Update Target Network to Online Network
            self.online_net, self.target_net = update_target_network(
                        sess,
                        self.online_net,
                        self.target_net)
           
            saver = tf.train.Saver(max_to_keep=None)

            #Reseting all tools
            self.reset_bookkeeping_tools()
            t = 0
            MAX_REWARD = 0


            for EPI in range(1, TRAIN_EPISODES+1):

                obs = self.env.reset(TRAIN=True)

                DONE, SOLVED = False, False

                while not DONE:

                    #Pick the decayed epsilon value
                    exploration = LinearDecay(
                        t, 
                        EPISODES_TO_EXPLORE*STEPS_PER_EPISODE, 
                        INITIAL_P, 
                        FINAL_P)

                    #Pick an action using online network
                    ACTION = choose_action(
                        obs, 
                        self.online_net, 
                        exploration,
                        self.env,
                        sess)

                    #Advance one step with the action in our environment
                    new_obs, _action, reward, DONE = self.env.step(ACTION)

                    #Add the Experience to the memory
                    replaybuffer.add(obs, _action, reward, new_obs, float(DONE))


                    obs = new_obs
                    t += 1

                    if t > self.UPDATE_FREQUENCY:
                        #Optimize online network with SGD
                        self.online_net, self.target_net = mini_batch_training(
                            sess,
                            self.env,
                            self.online_net,
                            self.target_net,
                            replaybuffer,
                            BATCH_SIZE,
                            GAMMA
                        )
                                            

                    if t % self.UPDATE_FREQUENCY == 0:
                        #Periodically copy online net to target net
                        self.online_net, self.target_net = update_target_network(
                            sess,
                            self.online_net,
                            self.target_net
                        )

                    if DONE:
                        '''End of Episode routines'''

                        #Close the Last Trade in portfolio if any
                        if self.env.portfolio.isHoldingTrade():
                            lastTime = self.env.sim.data.index[self.env.sim.curr_idx].to_pydatetime()
                            lastOpen = self.env.sim.data['Open'].iloc[self.env.sim.curr_idx]
                            self.env.portfolio.closeTrade(TIME=lastTime, OPEN=lastOpen)
                            
                        
                        #Update Bookkeeping Tools
                        AVERAGE_PIPS_PER_TRADE = self.env.portfolio.total_reward / self.env.portfolio.total_trades
                        self.journal_record.append(self.env.portfolio.journal)
                        self.avg_reward_record.append(AVERAGE_PIPS_PER_TRADE)
                        self.reward_record.append(self.env.portfolio.total_reward)
                        self.equity_curve_record.append(self.env.portfolio.equity_curve)
                        

                        #Print statements at the end of every statements
                        print("End of Episode %s, Total Reward is %s, Average Reward is %.3f"%(
                            EPI, 
                            self.env.portfolio.total_reward, 
                            AVERAGE_PIPS_PER_TRADE
                            ))
                        print("Percentage of time spent on exploring (Random Action): %s %%"%(
                            int(100 * exploration)))

                        #Save this score
                        if policy_measure == 'average':
                            SCORE = AVERAGE_PIPS_PER_TRADE
                        elif policy_measure == 'highest':
                            SCORE = self.reward_record[-1]
                        else:
                            SCORE = np.abs(AVERAGE_PIPS_PER_TRADE) * self.reward_record[-1]


                        #Is this SCORE greater than any current top 10s?
                        
                        TERMINAL_PATH = self.tensor_dir_template%EPI


                        #Condition: Only start recording this score if agent is no longer exploring 
                        if EPI > EPISODES_TO_EXPLORE:
                        
                            
                            if len(current_top_10s) < 10:
                                # Just append if there are not enough on the list
                                current_top_10s.append((EPI, SCORE))

                                #Sort the list
                                current_top_10s = sorted(current_top_10s, key=lambda x: x[1], reverse=True)

                                #Save the maximum score
                                MAX_REWARD = current_top_10s[0][1]

                                saver.save(sess, TERMINAL_PATH)

                            else:
                                
                                REPLACE = False
                                insertion_idx = None
                                for i, _tuple in enumerate(current_top_10s):
                                    _epi, _score = _tuple[0], _tuple[1]

                                    if SCORE > _score:
                                        MAX_REWARD = SCORE
                                        insertion_idx = i
                                        REPLACE = True
                                        break

                                #Remove from the last index, insert this
                                if REPLACE:
                                    current_top_10s.pop()
                                    current_top_10s.insert(insertion_idx, (EPI, SCORE))
                                    saver.save(sess, TERMINAL_PATH)

                        print ()

                        if exploration == FINAL_P and len(current_top_10s) == 10:
                            
                            if np.mean(self.reward_record[-16:-1]) > CONVERGENCE_THRESHOLD:
                                SOLVED = True

                        break

                if SOLVED:
                    print ("CONVERGED!")
                    print ()
                    break

        self.best_models = current_top_10s


    def trainSummary(self, TOP_N=3):
        
        #Plot Total Reward
        rewardPlot(self.reward_record, self.best_models, 'Total', TOP_N)

        #Plot Average Reward
        rewardPlot(self.avg_reward_record, self.best_models, "Average", TOP_N)
        
        for i,m in enumerate(self.best_models):
            eps = m[0] 
            print ("")
            print ("########   RANK {}   ###########".format(i+1))
            print ("Episode          | {}".format(eps))
            print ("Total Reward     | {0:.2f}".format(self.reward_record[eps-1]))
            print ("Average Reward   | {0:.2f}".format(self.avg_reward_record[eps-1]))
            print ("################################")
            print ("")


    def episodeReview(self, EPS):
        
        idx = EPS - 1

        journal = pd.DataFrame(self.journal_record[idx])

        buys = journal.loc[journal['Type']=='BUY', :]
        sells = journal.loc[journal['Type']=='SELL', :]

        print ("Summary Statistics for Episode %s \n"%(EPS))

        print ("Total Trades            | {}        (Buy){}       (Sell){} "\
            .format(journal.shape[0], buys.shape[0], sells.shape[0]))

        #Calculate Profit breakdown
        total_profit = journal.Profit.sum()
        buy_profit = buys.Profit.sum()
        sell_profit = sells.Profit.sum()

        print ("Profit (in pips)        | %.2f   (Buy)%.2f   (Sell)%.2f"\
            %(total_profit, buy_profit, sell_profit))

        #Calculate Win Ratio
        total_percent = (journal.loc[journal['Profit']>0,'Profit'].count()/ journal.shape[0]) * 100
        buy_percent = (buys.loc[buys['Profit']>0, 'Profit'].count()/buys.shape[0]) * 100
        sell_percent = (sells.loc[sells['Profit']>0, 'Profit'].count()/sells.shape[0]) * 100
        print ("Win Ratio               | %.2f%%    (Buy)%.2f%%   (Sell)%.2f %%"%(total_percent, buy_percent, sell_percent))

        duration = journal['Trade Duration'].mean()
        print ("Average Trade Duration  | %.2f"%(duration))

        #print candle_stick
        ohlcPlot(self.journal_record[idx], self.env.sim.data, self.equity_curve_record[idx])



    def test(self, EPISODE):
        '''
        EPISODE: int, episode to be selected
        '''
        assert len(os.listdir(self.PARENT_PATH)) > 0, "No saved tensor models are found for this model, please train the network"

        MODEL_PATH = self.tensor_dir_template%EPISODE

        with tf.Session() as sess:
            #Create restoration path
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_PATH)

            obs = self.env.reset(TRAIN=False)
            DONE= False

            while not DONE:
                
                #Select Action
                ACTION = choose_action(
                        obs, 
                        self.online_net, 
                        0, #Greedy selection
                        self.env,
                        sess)

                #Transit to next state given action
                new_obs, _, _, DONE = self.env.step(ACTION)
                
                obs = new_obs
            

            AVERAGE_PIPS_PER_TRADE = self.env.portfolio.total_reward / self.env.portfolio.total_trades
            self.journal_record.append(self.env.portfolio.journal)
            self.avg_reward_record.append(AVERAGE_PIPS_PER_TRADE)
            self.reward_record.append(self.env.portfolio.total_reward)
            self.equity_curve_record.append(self.env.portfolio.equity_curve)
            

            self.episodeReview(0)


    def liveTrading(self, MODEL_EPS, HISTORY=20, tradeFirst=False):
        '''
        Threaded implementation of listener and handler events
        MODEL_EPS: int, MODEL to be chosen
        '''
        #Set Environment and its portfolio to Live Mode
        self.env.setLive()

        #Clear up the portfolio
        self.env.portfolio.reset()

        #Initialize the time of the current incomplete candle
        #Set True if start trading on current candle, (Not Recommended during Market Close)
        if tradeFirst: 
            self.env.lastRecordedTime = None
        else:
            self.env.lastRecordedTime = self.env.api_Handle.getLatestTime(self.env.SYMBOL)

        MODEL_PATH = self.tensor_dir_template%MODEL_EPS

        #Tensorflow session with Chosen Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        #Initiate an event stack 
        events_q = LifoQueue(maxsize=1)

        listenerThread = Thread(target=self.env.candleListener, args=(events_q,))
        handlerThread = Thread(target=self.newCandleHandler, args=(events_q, sess))

        #Start threads
        listenerThread.start()
        handlerThread.start()


    def newCandleHandler(self, queue, SESS, HISTORY=20):
        '''
        Receives a new Candle event and perform action 
        '''
        
        while True:

            if not queue.empty():
                event = queue.get()

                if event == 'New Candle':

                    print ("Processing New Candle")

                    data, states = self.env.sim.build_data_and_states(HISTORY)
                    
                    ACTION = choose_action(
                        states[-1], #Latest state
                        self.online_net, 
                        0, #Greedy selection
                        self.env,
                        SESS)
                    
                    #Initiate position with Portfolio object
                    self.env.portfolio.newCandleHandler(ACTION)
                    
                    queue.task_done()
