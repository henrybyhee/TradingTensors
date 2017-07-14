import itertools

import gym


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from gym import error, spaces, utils
from gym.utils import seeding


class Portfolio(object):
    '''
    Parameters:
    trading_cost: Cost of taking a position is 3 pips by default
    _prices: Dataframe of Open and Close prices for reward calculation
    reward_normalizer: to convert precision of prices (0.0001) to number (1) for reward calculation
    total_reward: Keep track of reward accumulated in an episode
    current_time: the current index in the dataframe
    step: keep track of number of steps taken. Different from current_time

    curr_trade: A dictionary that records the details of an open position
    journal: Stores all records of curr_trade
    holding_trade: Boolean Flag to allow new position

    '''
    
    def __init__(self, prices, train_end_index, trade_period=1, max_price=10, denom=0.0001, cost=3):

        self.train_end = train_end_index
        self.trade_period = trade_period

        #Trading cost is 3 pips
        self.trading_cost = cost

        #Store list of Open price and Close price to manage reward calculation
        self._open = prices.Open.values
        self._close = prices.Close.values
        self._index = prices.index

        #To normalise reward terms
        self.reward_normalizer = 1./ denom
        self._reset()

    def _reset(self, train=True):
        self.total_reward = 0
        self.total_trades = 0
        self.average_profit_per_trade = 0
        self.equity_curve = []
        
        if train:
            self.current_time = 1
        else:
            self.current_time = self.train_end + 1
        

        self.curr_trade = {'Entry Price':0, 'Exit Price':0, 'Entry Time':None, 'Exit Time':None ,'Profit':0, 'Trade Duration':0, 'Type':None}
        self.journal=[]

        self.holding_trade = False


    def _reset_trade(self):
        self.curr_trade = {'Entry Price':0, 'Exit Price':0, 'Entry Time':None, 'Exit Time':None ,'Profit':0, 'Trade Duration':0, 'Type':None}


    def close_trade(self, curr_close_price, curr_time):
        if self.curr_trade['Type'] == 'SELL':

            #Update remaining keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            self.curr_trade['Profit'] = -1 *(curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost

            #Add the current trade to the journal
            self.journal.append(self.curr_trade)
            self._reset_trade()
            self.holding_trade = False
        
        if self.curr_trade['Type'] == 'BUY':
        #Action is 1, Selling to close the Long position
            
            #Update remaining  keys in curr_trade dict
            self.curr_trade['Exit Price'] = curr_close_price
            self.curr_trade['Exit Time'] = curr_time
            self.curr_trade['Profit'] = (curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost
           
            #Add curr_trade to journal, then reset curr_trade
            self.journal.append(self.curr_trade)
            self._reset_trade()
            
            self.holding_trade = False
    
            
    def _step(self, action):
        curr_open_price = self._open[self.current_time]
        curr_close_price = self._close[self.current_time]
        curr_time = self._index[self.current_time]
        prev_close_price =  self._close[self.current_time-1]
        reward = 0

        if self.holding_trade is False:
        # No open position at the moment

            if action == 0:
            # BUYING
                #Update keys on curr_trade
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "BUY"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward = (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost

                self.total_trades += 1
                self.holding_trade = True


            elif action == 1:
            # SELLING
                #Update keys on curr_trade
                self.curr_trade['Entry Price'] = curr_open_price
                self.curr_trade['Type'] = "SELL"
                self.curr_trade['Entry Time'] = curr_time
                self.curr_trade['Trade Duration'] += 1
                reward =  -1 * (curr_close_price - curr_open_price ) * self.reward_normalizer - self.trading_cost

                self.total_trades += 1
                self.holding_trade = True

        else:
        # Holding trade, Resolve Open position 
            if action == 2:
                self.curr_trade['Trade Duration'] += 1
                self.holding_trade = True

                if self.curr_trade['Type'] == 'SELL':
                    reward = -1 * (curr_close_price - prev_close_price) * self.reward_normalizer
                
                if self.curr_trade['Type'] == 'BUY':
                    reward = (curr_close_price - prev_close_price) * self.reward_normalizer

        #Closing trade once trade duration is reached
        if self.curr_trade['Trade Duration'] == self.trade_period:
            self.close_trade(curr_close_price, curr_time)
            
        
        self.total_reward += reward
        if self.total_trades > 0:
            self.average_profit_per_trade = self.total_reward / self.total_trades

        self.current_time += 1
        info = {'Average reward per trade': self.average_profit_per_trade}
        
        #Update equity at every step
        self.equity_curve.append(self.total_reward)

        return self.average_profit_per_trade, info


class Simulator(object):
    def __init__(self, csv, train_split, dummy_period=None, ATR=False, SMA=False, RSI=False, BB=False, train=True):
        df = pd.read_csv(csv, parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

        #Compute Returns based on consecutive closed prices 
        df['Return'] = df['Close'].pct_change()*100

        #Add indicators
        df = self._generate_indicators(df, ATR, SMA, RSI, BB)
        df = df[~np.isnan(df['Return'])].set_index('Date_Time')


        #Normalization of returns
        mean = df['Return'].mean()
        std = df['Return'].std()
        df['Return'] = ( df['Return'] - np.array(mean))/ np.array(std)

        #Generate future Closing price of n period
        if dummy_period is not None:
            for i in range(0, dummy_period+1):
                df['Close Price(n + %s)'%i] = df['Close'].shift(-i)
        
        
        ##Attributes
        self.data = df.dropna()
        self.date_time = self.data.index

        #Attributes related to the observation state: Return, ATR, 
        self.states = self.data.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).values
        
        #initialise labels for states
        self.state_labels = self.data.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).columns.tolist()

        self.count = self.states.shape[0]
        self.train_end_index = int(train_split * self.count)

        #update environment-related variables
        self.min_values = self.states.min(axis=0)
        self.max_values = self.states.max(axis=0)

        self._reset()


    def _generate_indicators(self, data, ATR, SMA, RSI, BB):
        _high = data.High.values
        _low = data.Low.values
        _close = data.Close.values

        if ATR:
            # Compute the ATR and perform Normalisation
            data['ATR'] = talib.ATR(_high, _low, _close, timeperiod=14)
            data['ATR'] = (data['ATR'] - np.nanmean(data['ATR']))/ np.std(data['ATR'])

        if SMA:
            # Compute the SMA and perform Normalisation
            data['SMA'] = talib.SMA(_close, timeperiod=14)
            data['SMA'] = (data['SMA'] - np.nanmean(data['SMA']))/ np.nanstd(data['SMA'])
        
        if RSI:
            data["RSI"] = talib.RSI(_close, timeperiod=14)
            data["RSI"] = (data["RSI"] - np.nanmean(data["RSI"]))/ np.nanstd(data["RSI"])
        
        if BB:
            bands = talib.BBANDS(_close, timeperiod=14)
            data["BBhigh"] = (bands[0] - np.nanmean(bands[0]))/ np.nanstd(bands[0])
            data["BBlow"] = (bands[1] - np.nanmean(bands[1]))/ np.nanstd(bands[1])

        data.dropna(inplace=True)
        
        return data

    def _reset(self, train=True):
        
        if train:
            self.current_index = 1
            self._end = self.train_end_index
        else:
            self.current_index = self.train_end_index + 1
            self._end = self.count - 1

        self._data = self.data.iloc[self.current_index:self._end+1]

    def _step(self):
        obs = self.states[self.current_index]
        self.current_index += 1
        
        done = self.current_index > self._end
        return obs, done

    #Adding intermarket forex prices to the environment
    def add_market(self, csv):
        
         data = pd.read_csv(csv, parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Intermarket Close', 'Volume'])
         data.set_index('Date_Time', inplace=True)

         close = data['Intermarket Close']
         close = self.data.join(close, how='left')['Intermarket Close'].fillna(0).values
         close = close.reshape(close.shape[0], 1)

         #Addition of variables to State, Update accordingly
         #Merge with df_index

         self.state_labels += ['Intermarket Close Price']
         self.states = np.column_stack((self.states,close))
         self.min_values = self.states.min(axis=0)
         self.max_values = self.states.max(axis=0)

class TradingEnv(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        print('Please invoke .initialise_simulator() method next to complete initialization')

    def initialise_simulator(self, csv, ATR, SMA, RSI, BB, trade_period=1, train_split= 0.8, dummy_period=None):
        self.sim = Simulator(csv, ATR=ATR, SMA=SMA, RSI=RSI, BB=BB, train_split=train_split, dummy_period=dummy_period)
        self.portfolio = Portfolio(prices=self.sim.data[['Open','Close']], trade_period= trade_period, train_end_index=self.sim.train_end_index)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values)
        
    def _step(self, action):
    #Return the observation, done, reward from Simulator and Portfolio
        obs, done = self.sim._step()
        reward, info = self.portfolio._step(action)

        return obs, reward, done, info
    
    def _reset(self, train=True):
        self.sim._reset(train)
        self.portfolio._reset(train)
    
    #Adds intermarket forex prices
    def add_market(self, csv):
        self.sim.add_market(csv)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values)

    def _render(self):
        print("Rendering")
        pass
