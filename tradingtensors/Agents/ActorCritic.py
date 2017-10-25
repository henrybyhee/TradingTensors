#Implementation is very similar to MorvanZhou's, read more of his work here https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_discrete_action.py#L147

import tensorflow as tf
from tensorflow import layers
import numpy as np


GLOBAL_REWARDS = []



#Architecture of the Actor-Critic Network
class Brain(object):
    def __init__(self, actions, states, name, Global_Net=None, isGlobal=False, COEF_REG=0.001, shared=False):

        self.n_actions = actions #Number of discrete actions in environment
        self.n_states = states
        self.shared = shared

        if isGlobal:
            with tf.variable_scope(name):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
                self.actor_output, self.critic_output, self.actor_params, self.critic_params = self.build_net(name)

        else:
            #Creating Local Nets

            with tf.variable_scope(name):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name='s')
                self.actor_output, self.critic_output, self.actor_params, self.critic_params = self.build_net(name)

                #Additional Resources to compute Loss function
                self.action_history = tf.placeholder(tf.int32, [None], name='action_history')
                self.target_value = tf.placeholder(tf.float32, [None, 1], name='target_value')

                self.advantage = self.target_value - self.critic_output #TD-Error

                #Optimizer and decay learning rate
                _global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(0.001, _global_step, 10000, 0.96, staircase=True)
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate)


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
                    self.actor_grads = tf.gradients(self.policy_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.value_loss, self.critic_params)

                
            #Pushing Operations: Apply local gradients to Global Net
            self.push_actor_params_op = self.optimizer.apply_gradients(zip(self.actor_grads, Global_Net.actor_params), global_step=_global_step)
            self.push_critic_params_op = self.optimizer.apply_gradients(zip(self.critic_grads, Global_Net.critic_params), global_step=_global_step)

            #Pulling Operations: Copy Global Net to Local Net
            self.pull_actor_params_op = [ local_param.assign(global_param) for local_param, global_param in zip(self.actor_params, Global_Net.actor_params)]
            self.pull_critic_params_op = [ local_param.assign(global_param) for local_param, global_param in zip(self.critic_params, Global_Net.critic_params)]


    def build_net(self, scope, shared=False):
        init = tf.random_uniform_initializer(0.,1.)

        if shared:
            #Shared layer 
            with tf.variable_scope('shared'):

                hidden = layers.dense(self.s, 128, tf.nn.tanh, kernel_initializer=init, name='shared_layer')
                hidden = layers.dense(hidden, 64, tf.nn.relu6, kernel_initializer=init, name='shared_layer')

                with tf.variable_scope('Actor'):
                    a_output = layers.dense(hidden, self.n_actions, tf.nn.softmax, kernel_initializer=init, name='a_out')

                with tf.variable_scope('Critic'):
                    c_output = layers.dense(hidden, 1, kernel_initializer=init, name='c_out')

            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope + 'shared/Actor')
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope + 'shared/Critic')

        else:
            # Separate Actor and Critic Network
            with tf.variable_scope('Actor'):
                a_hidden = layers.dense(self.s, 64, tf.nn.relu6, kernel_initializer=init, name='a_hidden')
                a_output = layers.dense(a_hidden, self.n_actions, tf.nn.softmax, kernel_initializer=init, name='a_out')

            with tf.variable_scope('Critic'):
                c_hidden = layers.dense(self.s, 128, tf.nn.tanh, kernel_initializer=init, name='c_hidden')
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

        sess.run([self.push_actor_params_op, self.push_critic_params_op], feed_dict=feed_dict)

    def pull_from_global_network(self, sess):
        sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, sess, s):

        policy = sess.run(self.actor_output, feed_dict={
            self.s: s[np.newaxis,:] 
        })

        action = np.random.choice(range(self.n_actions), p=policy.ravel())

        return action

def discount_reward(r, v_tp1):
    #Resolve credit assignment problem, r is array of rewards, v_s is the value estimate from the next state
    GAMMA =0.9

    discounted_r = np.zeros_like(r)

    for t in reversed(range(0, len(r))):
        v_tp1 = r[t] + GAMMA * v_tp1
        discounted_r[t] = v_tp1

    return discounted_r


class Agent(object):
    def __init__(self, name, env_create, global_network, maximum_episodes=500, trading=True):
        self.env = env_create()

        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.name = name
        self.local_brain = Brain(actions=num_actions, states=num_states, name=self.name, Global_Net=global_network)

        self.trading = trading

        self.max_episodes = maximum_episodes

        

    def work(self, coord, sess, rew_threshold, update_freq=15, Oanda=False, train=True):

        steps = 1
        epi = 1

        
            
        while not coord.should_stop():
            global GLOBAL_REWARDS

            epi_reward = 0
            
            #Store the transition phase
            memory_s, memory_a, memory_r  = [], [], []
            
            #Reset Environment and obtain initial state
            
            if self.trading:
                #For Trading env only
                self.env._reset(train=True, Oanda=self.env.Oanda)
                s = self.env.sim.states[0]
            
            else:
                #Other OpenAI environment
                s = self.env.reset()

            action_dict = {i:0 for i in range(self.local_brain.n_actions)}

            while True:
                
                a = self.local_brain.choose_action(sess, s)

                
                if self.trading:
                    if self.env.portfolio.holding_trade:
                        a = 2

                    s_, r, done, _ = self.env._step(a)
                
                else:
                    s_, r, done, _ = self.env.step(a)

                action_dict[a] += 1

                epi_reward += r

                #Store the transition
                memory_s.append(s)
                memory_a.append(a)
                memory_r.append(r)

                if done or (steps % update_freq == 0):
                    
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
                    
                    if self.trading:
                        #Use portfolio reward tracker for greater accuracy
                        epi_reward = self.env.portfolio.total_reward

                    print ("%s, Episode:%s, Reward: %s"%(self.name, epi, epi_reward))
                    print (action_dict)
                    
                    GLOBAL_REWARDS.append(epi_reward)

                    epi += 1

                    #Stop the training once either reward or episode limit is reached
                    reward_reached = np.mean(GLOBAL_REWARDS[-10:]) >= rew_threshold
                    episode_reached = epi >= self.max_episodes

                    if reward_reached or episode_reached:
                        coord.request_stop()

                    break
        










