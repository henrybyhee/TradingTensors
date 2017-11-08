#Implementation is very similar to MorvanZhou's, read more of his work here https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_discrete_action.py#L147

import datetime as dt
import os
from threading import Thread
from queue import LifoQueue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import layers

from ..settings.A3Csettings import (ACTOR_LAYER, COEF_REG, CRITIC_LAYER, GAMMA,
                                    L2_REG_LAMBDA, LEARNING_RATE, NUM_WORKERS,
                                    SHARED, UPDATE_FREQ, SHARED_LAYER)
from .visual_utils import ohlcPlot

GLOBAL_REWARDS = []



#Architecture of the Actor-Critic Network
class Brain(object):
    def __init__(self, states, name, Global_Net=None, isGlobal=False):

        self.n_actions = 3 #Number of discrete actions in environment
        self.n_states = states
        
        if isGlobal:
            with tf.variable_scope(name):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')

                if SHARED:
                    self.actor_output, self.critic_output, self.params = self.build_net(name)
                else:
                    self.actor_output, self.critic_output, self.actor_params, self.critic_params = self.build_net(name)

        else:
            #Creating Local Nets

            with tf.variable_scope(name):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')

                if SHARED:
                    self.actor_output, self.critic_output, self.params = self.build_net(name)
                else:
                    self.actor_output, self.critic_output, self.actor_params, self.critic_params = self.build_net(name)

                #Additional Resources to compute Loss function
                self.action_history = tf.placeholder(tf.int32, [None], name='action_history')
                self.target_value = tf.placeholder(tf.float32, [None, 1], name='target_value')

                self.advantage = self.target_value - self.critic_output #TD-Error

                #Optimizer and decay learning rate
                _global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(LEARNING_RATE, _global_step, 10000, 0.96, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate)


                with tf.variable_scope('value_loss'):
                    self.value_loss = tf.reduce_mean(tf.square(self.advantage))

                with tf.variable_scope('policy_loss'):

                    # Compute PG Loss
                    action_mask = tf.one_hot(self.action_history, self.n_actions, on_value=1.0, off_value=0.0)
                    self.prob_under_policy = tf.reduce_sum(self.actor_output * action_mask, axis=1, keep_dims=True)
                    self.neglogp = - tf.log(self.prob_under_policy + 1e-13)
                    self.actor_loss = tf.stop_gradient(self.advantage) * self.neglogp

                    #Compute Entropy
                    self.entropy = - tf.reduce_sum(self.actor_output * - tf.log(self.actor_output + 1e-13), axis=1, keep_dims=True)
                    
                    #Total Policy Loss: Entropy * coefficient_regularization + pg_loss
                    self.policy_loss = tf.reduce_mean(COEF_REG* self.entropy + self.actor_loss)


                with tf.variable_scope('gradients'):
                    if SHARED:
                        self.total_loss = self.policy_loss + self.value_loss
                        self.grads = tf.gradients(self.total_loss, self.params)
                    else: 
                        self.actor_grads = tf.gradients(self.policy_loss, self.actor_params)
                        self.critic_grads = tf.gradients(self.value_loss, self.critic_params)

            if SHARED:
                #Pushing Operations: Apply local gradients to Global Net
                self.push_params_op = self.optimizer.apply_gradients(zip(self.grads, Global_Net.params), global_step=_global_step)

                #Pulling Operations: Copy Global net to Local Net
                self.pull_params_op = [ local_param.assign(global_param) for local_param, global_param in zip(self.params, Global_Net.params)]

            else:
                #Pushing Operations: Apply local gradients to Global Net
                self.push_actor_params_op = self.optimizer.apply_gradients(zip(self.actor_grads, Global_Net.actor_params), global_step=_global_step)
                self.push_critic_params_op = self.optimizer.apply_gradients(zip(self.critic_grads, Global_Net.critic_params), global_step=_global_step)

                #Pulling Operations: Copy Global Net to Local Net
                self.pull_actor_params_op = [ local_param.assign(global_param) for local_param, global_param in zip(self.actor_params, Global_Net.actor_params)]
                self.pull_critic_params_op = [ local_param.assign(global_param) for local_param, global_param in zip(self.critic_params, Global_Net.critic_params)]


    def build_net(self, scope):
        init = tf.random_uniform_initializer(0.,1.)

        if SHARED:
            #Shared layer 
            
            with tf.variable_scope('shared'):

                for i, layer in enumerate(SHARED_LAYER):
                    hidden = layers.dense(
                        self.s, layer, tf.nn.tanh, 
                        kernel_initializer=init, 
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_REG_LAMBDA),
                        name='layer_%s'%(i+1)
                        )
                
                with tf.variable_scope('Actor'):
                    a_output = layers.dense(hidden, self.n_actions, tf.nn.softmax, kernel_initializer=init, name='a_out')

                with tf.variable_scope('Critic'):
                    c_output = layers.dense(hidden, 1, kernel_initializer=init, name='c_out')
            
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope + '/shared')

            return a_output, c_output, params

        else:
            # Separate Actor and Critic Network
            with tf.variable_scope('Actor'):
                
                #Create Actor Network
                for layer in ACTOR_LAYER:
                    a_hidden = layers.dense(
                        self.s, layer, 
                        tf.nn.relu6, 
                        kernel_initializer=init, 
                        name='a_hidden', 
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_REG_LAMBDA))
                
                a_output = layers.dense(a_hidden, self.n_actions, tf.nn.softmax, kernel_initializer=init, name='a_out')

            with tf.variable_scope('Critic'):
                
                #Create Critic Network
                for layer in CRITIC_LAYER:
                    c_hidden = layers.dense(
                        self.s, 128, 
                        tf.nn.tanh, 
                        kernel_initializer=init, 
                        name='c_hidden', 
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_REG_LAMBDA))
                
                c_output = layers.dense(c_hidden, 1, kernel_initializer=init, name='c_out')
            
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope + '/Actor')
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope + '/Critic')

            return a_output, c_output, actor_params, critic_params

    def push_to_global_network(self, sess, feed_s, feed_a, feed_v):
        #Apply local gradients to the global network

        feed_dict = {
            self.s : feed_s,
            self.action_history: feed_a,
            self.target_value: feed_v
        }

        if SHARED:
            sess.run(self.push_params_op, feed_dict=feed_dict)
        else:    
            sess.run([self.push_actor_params_op, self.push_critic_params_op], feed_dict=feed_dict)


    def pull_from_global_network(self, sess):
        #Copy global network to local networks 

        if SHARED:
            sess.run(self.pull_params_op)
        else:
            sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, sess, s):

        policy = sess.run(self.actor_output, feed_dict={
            self.s: s[np.newaxis,:] 
        })

        action = np.random.choice(range(self.n_actions), p=policy.ravel())

        return action

def discount_reward(r, v_tp1):
    #Resolve credit assignment problem, r is array of rewards, v_s is the value estimate from the next state
    discounted_r = np.zeros_like(r)

    for t in reversed(range(0, len(r))):
        #Adding v_tp1 * gamma to every rewards in each steps
        v_tp1 = r[t] + GAMMA * v_tp1
        discounted_r[t] = v_tp1

    return discounted_r


class Agent(object):
    def __init__(self, name, env, global_network):
        
        self.env = env()
        
        #Define states dimension
        num_states = self.env.observation_space

        self.name = name
        self.local_brain = Brain(states=num_states, name=self.name, Global_Net=global_network)

    def work(
        self, coord, sess, 
        rew_threshold,  MAX_EPISODES=500):

        steps = 1
        eps = 1

        while not coord.should_stop():
            global GLOBAL_REWARDS

            eps_reward = 0
            
            #Store the transition phase
            memory_s, memory_a, memory_r  = [], [], []
            
            #Reset Environment and obtain initial state
            s = self.env.reset(TRAIN=True)

            while True:
                
                a = self.local_brain.choose_action(sess, s)

                s_, a, r, done = self.env.step(a)
                
                eps_reward += r

                #Store the transition
                memory_s.append(s)
                memory_a.append(a)
                memory_r.append(r)

                if done or (steps % UPDATE_FREQ == 0):
                    
                    #Estimate the Value Function from next state onwards
                    if done:
                        v_tp1 = 0 #Terminal State, set to zero
                    else:
                        v_tp1 = sess.run(self.local_brain.critic_output, feed_dict={self.local_brain.s: s_[np.newaxis,:]}) #Bootstrapped for non-terminal states

                    
                    #Apply discount factors to reward
                    value_targets = discount_reward(memory_r, v_tp1)


                    feed_s, feed_a, feed_v = np.vstack(memory_s), np.array(memory_a), np.vstack(value_targets)
                    
                    #Apply local gradients onto global network (Actor and Critic)
                    self.local_brain.push_to_global_network(sess, feed_s, feed_a, feed_v)

                    #Use the updated params from global network
                    self.local_brain.pull_from_global_network(sess)

                    memory_s, memory_a, memory_r = [], [], []
                
                #Important!
                s = s_
                steps += 1

                if done:
                    if self.env.portfolio.isHoldingTrade():
                        lastTime = self.env.sim.data.index[self.env.sim.curr_idx].to_pydatetime()
                        lastOpen = self.env.sim.data['Open'].iloc[self.env.sim.curr_idx]
                        self.env.portfolio.closeTrade(TIME=lastTime, OPEN=lastOpen)

                    #Use portfolio reward tracker for greater accuracy
                    eps_reward = self.env.portfolio.total_reward

                    print ("%s, Episode:%s, Reward: %s"%(self.name, eps, eps_reward))

                    GLOBAL_REWARDS.append(eps_reward)

                    eps += 1

                    #Stop the training once either reward or episode limit is reached
                    reward_reached = np.mean(GLOBAL_REWARDS[-10:]) >= rew_threshold
                    episode_reached = eps >= MAX_EPISODES

                    if reward_reached or episode_reached:
                        coord.request_stop()

                    break
        

class A3CAgent(object):

    def __init__(self, create_env, PATH):
        
        self.env_init = create_env
        self.env = self.env_init()
    
        self.global_net = Brain(
            self.env.observation_space,
            name='Global',
            isGlobal=True
            )
        self.workers = self.create_workers()
        self.PATH = PATH
        self.latest_model = os.path.join(self.PATH, 'latest.ckpt')

    def create_workers(self):
        workers = []

        for i in range(NUM_WORKERS):
            name = "Worker_%s"%(i)

            #Create Agent
            worker = Agent(name, self.env_init, self.global_net)

            workers.append(worker)
        return workers


    def trainGlobalNet(self, REWARD_THRESHOLD, MAX_EPS):
        
        sess= tf.Session()

        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        tasks = []

        for worker in self.workers:

            job = lambda: worker.work(
                coord, sess, 
                REWARD_THRESHOLD,
                MAX_EPS
                )
            task = Thread(target=job)
            task.start()
            tasks.append(task)

        #Wait for all threads to finish training
        coord.join(tasks)


        saver.save(sess, self.latest_model)
        
        sess.close()


    def trainSummary(self):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.title('Rewards')
        ax.plot(GLOBAL_REWARDS)
        plt.show()

    
    def performs(self, TRAIN):
    
        sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(sess, self.latest_model)

        obs = self.env.reset(TRAIN=TRAIN)
        DONE = False

        while not DONE:

            ACTION = self.global_net.choose_action(sess, obs)

            next_obs, _, _, DONE = self.env.step(ACTION)

            obs = next_obs

        if self.env.portfolio.isHoldingTrade():
            lastTime = self.env.sim.data.index[self.env.sim.curr_idx].to_pydatetime()
            lastOpen = self.env.sim.data['Open'].iloc[self.env.sim.curr_idx]
            self.env.portfolio.closeTrade(TIME=lastTime, OPEN=lastOpen)


        AVERAGE_PIPS_PER_TRADE = self.env.portfolio.total_reward / self.env.portfolio.total_trades
        self.journal_record = self.env.portfolio.journal
        self.avg_reward_record = AVERAGE_PIPS_PER_TRADE
        self.reward_record =self.env.portfolio.total_reward
        self.equity_curve_record = self.env.portfolio.equity_curve
        
        self.summaryPlot()

    def summaryPlot(self):
        '''
        Review of the Agent's Performance
        '''

        journal = pd.DataFrame(self.journal_record)

        buys = journal.loc[journal['Type']=='BUY', :]
        sells = journal.loc[journal['Type']=='SELL', :]
        
        print ("Summary Statistics (Test)\n")

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
        
        #Duration
        duration = journal['Trade Duration'].mean()
        print ("Average Trade Duration  | %.2f"%(duration))


        ohlcPlot(self.journal_record, self.env.sim.data, self.equity_curve_record)


    def liveTrading(self, HISTORY=20, tradeFirst=False):

        self.env.setLive()

        self.env.portfolio.reset()

        #Initialize the time of the current incomplete candle
        #Set True if start trading on current candle, (Not Recommended during Market Close)
        if tradeFirst: 
            self.env.lastRecordedTime = None
        else:
            self.env.lastRecordedTime = self.env.api_Handle.getLatestTime(self.env.SYMBOL)

        #Tensorflow session with Chosen Model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.latest_model)

        #Initiate an event stack 
        events_q = LifoQueue(maxsize=1)

        listenerThread = Thread(target=self.env.candleListener, args=(events_q,))
        handlerThread = Thread(target=self.newCandleHandler, args=(events_q, sess))

        #Start threads
        listenerThread.start()
        handlerThread.start()


    def newCandleHandler(self, queue, SESS, HISTORY=20):
        '''
        Function that retrieves data from Oanda, generate states and open position
        '''
        while True:

            if not queue.empty():
                event = queue.get()

                if event == 'New Candle':
                    
                    #Generate state from candle
                    data, states = self.env.sim.build_data_and_states(HISTORY)

                    ACTION = self.global_net.choose_action(SESS, states[-1])

                    #Initiate position with Porfolio object
                    self.env.portfolio.newCandleHandler(ACTION)


                    if ACTION == 2:
                        print ("No action taken, Agent idles")
                    queue.task_done()
