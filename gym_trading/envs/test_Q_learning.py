import gym
import gym_trading
from gym_trading.envs.Q_learning import Q


csv = r"C:\Users\Henry\Documents\Python Stuff\Upwork Trial\Forex Ai\GBPUSD240.csv"
env = gym.make('trading-v0')
env.initialise_simulator(csv, ATR=True, SMA=True, RSI=True, BB=True, trade_period=5, train_split=0.7, dummy_period=None)


'''If another market is to be added, execute these functions
#csv = 'another_market.csv'
#env.add_market(csv)
'''

Q_learning = Q(env, train_episodes=1000, learning_rate=0.2, gamma=0.9)

Q_learning.train()

#Summary of Training
Q_learning._overall_summary()


#Begin testing
Q_learning.test(100)

#Summary of testing
Q_learning._overall_summary()

#Episode review
x = input("Enter the episode to be reviewed: \n")
Q_learning.episode_review(x)


