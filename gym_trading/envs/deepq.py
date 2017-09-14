import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


def huber_loss(error, delta=1.0):
    return tf.where(
        tf.abs(error) < delta,
        tf.square(error) * 0.5,
        delta * (tf.abs(error) - 0.5 * delta)
    )


class DQN(object):
    def __init__(self, env, hiddens, scope):
        
        self.num_actions = env.action_space.n
        self.inpt_dim = env.observation_space.shape[0]
 
        with tf.variable_scope(scope):
            self._inputs = tf.placeholder(tf.float32, [None, self.inpt_dim])
            
            self.build_q_network(hiddens)
            
            self.create_optimizer()
            
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            
    def build_q_network(self, hiddens):

        out = self._inputs
        
        for hidden in hiddens:
            out= layers.fully_connected(inputs=out, num_outputs= hidden, activation_fn=tf.tanh)
            
        self.Q_t = layers.fully_connected(out, self.num_actions,  activation_fn=None)
        self.Q_action = tf.argmax(self.Q_t, dimension=1)
        
        
    def create_optimizer(self):
        #Placeholder to hold values for Q_values estimated by target_network
        self.target_q_t = tf.placeholder(tf.float32, [None])
        
        #Compute current_Q estimation using online network, states and action are drawn from training batch 
        self.action = tf.placeholder(tf.int64, [None])
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0)
        self.current_Q = tf.reduce_sum(self.Q_t * action_one_hot, reduction_indices=1)
        

        #Difference between target_network and online network estimation
        self.td_error = huber_loss(self.target_q_t - self.current_Q)

        self.loss = tf.reduce_mean(self.td_error)

        #Dynamic Learning steps- decaying with episodes
        global_step = tf.Variable(0, trainable=False)
        learner_decay = tf.train.exponential_decay(1e-3, global_step, 10000, 0.96, staircase=True)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learner_decay)
        self.optimize = self.trainer.minimize(self.loss, global_step=global_step)


def mini_batch_training(session, env, online, target, replaybuff, BATCH_SIZE=32, discount=0.99):
    '''
    Sample Batch from memory and optimize online network
    '''
    obses_t, actions, rewards, obses_tp1, terminal = replaybuff.sample(BATCH_SIZE)
    
    #Double Q learning implementation
    state_shape = [BATCH_SIZE, env.observation_space.shape[0]]
    
    #Use online network to generate next actions
    next_action = session.run(online.Q_action, feed_dict ={
        online._inputs: np.reshape(obses_tp1, state_shape)
    })
    #Use target network to predict next Q_value
    next_Q = session.run(target.Q_t, feed_dict ={
            target._inputs: np.reshape(obses_tp1, state_shape)
        })
    
    #Select Q_values indexed by pred_actions
    Q_prime = [next_Q[i][a] for i, a in enumerate(next_action)]
    

    #Update Rule of the Bellman Equation
    target_q_t = rewards + (1. - terminal) * discount * Q_prime
    
    _ = session.run([online.optimize], feed_dict={
        online.target_q_t: target_q_t,
        online.action: actions, 
        online._inputs: np.reshape(obses_t, state_shape)
    })
    
    return online, target


def q_act(state, online, EPSILON, env, session):
    shape = [1, env.observation_space.shape[0]]
    if np.random.random() < EPSILON:
        #Exploration 
        action = np.random.choice(env.action_space.n)
    else:
        #Exploitation
        action= session.run(online.Q_action, feed_dict={online._inputs: np.reshape(state,  shape)})[0]
    return action


def update_target_network(session, online, target):
    #Copy variables of online network to target network 
    online_vars = online.variables
    target_vars = target.variables
    for on_, tar_ in zip(online_vars, target_vars):
        session.run(tf.assign(tar_,on_))
    return online, target



