import itertools

import gym
import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from gym import error, spaces, utils
from gym.utils import seeding
from .astro_coordinates import get_planet_coordinates
from copy import deepcopy
import time
from datetime import datetime

import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
from requests.exceptions import ConnectionError, ChunkedEncodingError


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
        
        self.Oanda = False
        self.client = None 
        self.accountID = None
        self.data4order = None
        self.instrument = None
        self.stream_prices = None
    
    def set_client(self, client, accountID, data4order, instrument):
        self.Oanda = True
        self.client = client
        self.accountID = accountID
        self.data4order = data4order
        self.instrument = instrument
        
        # params = {"instruments": self.instrument}
        # r = pricing.PricingStream(accountID=accountID, params=params)
        # self.stream_prices = self.client.request(r)
        
    def _reset(self, train=True, Oanda=False):
        self.total_reward = 0
        self.total_trades = 0
        self.average_profit_per_trade = 0
        self.equity_curve = []
        self.Oanda = Oanda
        
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
            if (self.Oanda):
                
                data = { "shortUnits": "ALL", }
                r = positions.PositionClose(accountID=self.accountID,
                             instrument=self.instrument,
                             data=data)
                
                while (True):
                    try:
                        self.client.request(r)
                        break
                    except ConnectionError:
                        continue
                
                 # save data from r.response 
                self.curr_trade['Exit Price'] = r.response["shortOrderFillTransaction"]["price"]
                self.curr_trade['Exit Time'] = r.response["shortOrderFillTransaction"]["time"]
                self.curr_trade['Profit'] = float(r.response["shortOrderFillTransaction"]["pl"]) * self.reward_normalizer
                print ("Close order")
                print ("Exit Price is %s" % self.curr_trade['Exit Price'] )
                print ("Exit Time is %s" % self.curr_trade['Exit Time'] )
                print ("Profit is %s" % self.curr_trade['Profit'])

            else:

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
            if (self.Oanda):
                data = { "longUnits": "ALL",}
                
                
                r = positions.PositionClose(accountID=self.accountID,
                             instrument=self.instrument,
                             data=data)
                while (True):
                    try:
                        self.client.request(r)
                        break
                    except ConnectionError:
                        continue
                

                # save data from r.response
                self.curr_trade['Exit Price'] = r.response["longOrderFillTransaction"]["price"]
                self.curr_trade['Exit Time'] = r.response["longOrderFillTransaction"]["time"]
                self.curr_trade['Profit'] = float(r.response["longOrderFillTransaction"]["pl"]) * self.reward_normalizer
                
                print ("Close order")
                print ("Exit Price is %s" % self.curr_trade['Exit Price'] )
                print ("Exit Time is %s" % self.curr_trade['Exit Time'] )
                print ("Profit is %s" % self.curr_trade['Profit'])
            else:            
				#Update remaining  keys in curr_trade dict
                self.curr_trade['Exit Price'] = curr_close_price
                self.curr_trade['Exit Time'] = curr_time
                self.curr_trade['Profit'] = (curr_close_price - self.curr_trade['Entry Price']) * self.reward_normalizer - self.trading_cost
			   
		    #Add curr_trade to journal, then reset curr_trade
            self.journal.append(self.curr_trade)
            self._reset_trade()
            self.holding_trade = False
    
            
    def _step(self, action):
        if (self.Oanda):
            curr_open_price = 0
            curr_close_price = 0
            curr_time = None
            prev_close_price = 0
        else:
            curr_open_price = self._open[self.current_time]
            curr_close_price = self._close[self.current_time]
            curr_time = self._index[self.current_time]
            prev_close_price =  self._close[self.current_time-1]
        
        reward = 0

        if self.holding_trade is False:
        # No open position at the moment

            if action == 0:
            # BUYING
                if (self.Oanda):
                    data = deepcopy(self.data4order)
                    
                    data["order"]["instrument"] = self.instrument
                    order_cr = orders.OrderCreate(self.accountID, data=data)
                    

                    while (True):
                        success = False
                        while(not success):
                            try:
                                params = {"instruments": self.instrument}
                                pr_info = pricing.PricingInfo(accountID=self.accountID, params=params) # self.stream_prices.__next__()["bids"][0]["price"]
                                rv = self.client.request(pr_info)
                                data["order"]["price"] = pr_info.response["prices"][0]["bids"][0]["price"]

                                success = True
                            except KeyError: 
                                success = False
                            except ChunkedEncodingError:
                                success = False


                        try:
                            self.client.request(order_cr)
                            break
                        except ConnectionError:
                            continue
                    
                    
                    
                    self.curr_trade['Entry Price'] = order_cr.response["orderFillTransaction"]["price"]
                    self.curr_trade['Entry Time'] = order_cr.response["orderFillTransaction"]["time"]
                    
                    print ("Open LONG")
                    print ("Price is %s" % self.curr_trade['Entry Price'] )
                    print ("Time is %s" % self.curr_trade['Entry Time']  )
                    
                else:
					#Update keys on curr_trade
                    self.curr_trade['Entry Price'] = curr_open_price
                    self.curr_trade['Entry Time'] = curr_time
                    reward = (curr_close_price - curr_open_price) * self.reward_normalizer - self.trading_cost
					         
                self.curr_trade['Type'] = "BUY"
                self.curr_trade['Trade Duration'] += 1
                self.total_trades += 1
                self.holding_trade = True


            elif action == 1:
            # SELLING
                if (self.Oanda):
                    data = deepcopy(self.data4order)
     
                    data["order"]["units"] = -1 
                    data["order"]["instrument"] = self.instrument
                    order_cr = orders.OrderCreate(self.accountID, data=data)
                    
                    while (True):
                        success = False
                        while(not success):
                            try:
                                # data["order"]["price"] = self.stream_prices.__next__()["asks"][0]["price"]
                                params = {"instruments": self.instrument}
                                pr_info = pricing.PricingInfo(accountID=self.accountID, params=params) # self.stream_prices.__next__()["bids"][0]["price"]
                                rv = self.client.request(pr_info)
                                data["order"]["price"] = pr_info.response["prices"][0]["asks"][0]["price"]
                                success = True
                            except KeyError: 
                                success = False
                            except ChunkedEncodingError:
                                success = False


                        try:
                            self.client.request(order_cr)
                            break
                        except ConnectionError:
                            continue
                    
                    self.curr_trade['Entry Price'] = order_cr.response["orderFillTransaction"]["price"]
                    self.curr_trade['Entry Time'] = order_cr.response["orderFillTransaction"]["time"]
                    print ("Open SHORT")
                    print ("Price is %s" % self.curr_trade['Entry Price'] )
                    print ("Time is %s" % self.curr_trade['Entry Time']  )
                
                else:
					#Update keys on curr_trade
                    self.curr_trade['Entry Price'] = curr_open_price
                    self.curr_trade['Entry Time'] = curr_time

                    reward =  -1 * (curr_close_price - curr_open_price ) * self.reward_normalizer - self.trading_cost
				
				
                self.curr_trade['Type'] = "SELL"
                self.curr_trade['Trade Duration'] += 1
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
    def __init__(self, instrument, additional_pairs, accountID, access_token, input_planets_data={}, n_periods=1,  time_frame="M1", oanda=False, size_of_history=200, dummy_period=None, ATR=False, SMA=False, RSI=False, BB=False, train_split=0.8, train=True):
        
        self.instrument = instrument
        self.additional_pairs = additional_pairs
        
        self.Oanda = oanda
        self.size_of_history = size_of_history
        self.n_periods = n_periods
        self.time_frame = time_frame
        self.ATR=ATR
        self.SMA=SMA
        self.RSI=RSI
        self.BB=BB
        self.input_planets_data = input_planets_data
        self.n_periods = n_periods
        
        self.accountID = accountID
        self.access_token = access_token
        self.client = None
        
        self._form_data(train_split=train_split, dummy_period=dummy_period)
        
        self._reset(train, self.Oanda)
        
    def _form_data(self, mean_std=True, train_split=None, dummy_period=None):
        
        while (True):
            if (self.Oanda): 
                # get data from Oanda
                self.client = oandapyV20.API(access_token=self.access_token)
                df = pd.DataFrame(self._get_historical_data_from_Oanda(self.instrument, self.size_of_history))
            else:
                # get from csv file
                csv = self.instrument + self.time_frame + ".csv"
                df = pd.read_csv(csv, parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])
            df.set_index(["Date_Time"], inplace=True, verify_integrity=False)
            df['Return'] = df['Close'].pct_change()*100
            
            
            df = self._generate_indicators(df, self.ATR, self.SMA, self.RSI, self.BB)
            df.dropna(axis=0,  inplace=True)
            
            self.delta = (df.index[1] - df.index[0]).to_pytimedelta()
            
            for pair in self.additional_pairs:
                if (self.Oanda):
                    df2 = pd.DataFrame(self._get_historical_data_from_Oanda(pair, self.size_of_history))
                else:
                    csv = pair + self.time_frame + ".csv"
                    df2 = pd.read_csv(csv, parse_dates=[[0,1]], header=None, names=['Date','Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
                df2["Date_Time"] = pd.to_datetime(df2["Date_Time"])
                df2.set_index(["Date_Time"], inplace=True, verify_integrity=False)
    
    
                df2[pair] = df2['Close'].pct_change()*100
                df2.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
                
                
                df = df.join(df2, how='inner')
            
          
            # add planets coordinates
            dates = [date_time.to_pydatetime() for date_time in df.index]
            
           
            
            planets = get_planet_coordinates(dates, self.input_planets_data, self.delta, n_periods=self.n_periods)
            df = df.join(planets, how='inner')
            
            
            df.dropna(axis=0,  inplace=True)
            
            if (len(df) > 1):
                break
        
        
        if mean_std:
            self.mean = df.mean()
            self.std = df.std()
            
            
        #########################
        # Normalization of data
        indx = np.logical_not(df.columns.isin(['Open', 'Close', 'High', 'Low', 'Volume']))

        df.loc[:, indx] = ( df.loc[:, indx] - self.mean.loc[indx]) / self.std.loc[indx]
        
        ##Attributes
        self.data = df
        self.date_time = df.index
        self.count = df.shape[0]

        if (train_split is None):
            self.train_end_index = -1
        else:
            self.train_end_index = int(train_split * self.count)
    
        #Attributes related to the observation state: Return
        
        self.states = self.data.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).values
        self.state_labels = self.data.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).columns.tolist()
        self.min_values = df.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).min(axis=0).values
        self.max_values = df.drop(['Open', 'Close', 'High', 'Low', 'Volume'], axis=1).max(axis=0).values
        

        self.last_time = self.data.index[-1].to_pydatetime()
        
        
        
        #Generate previous Close
        if dummy_period is not None:
    
            close_prices = pd.DataFrame()
            close_prices['Close'] = self.data["Close"]
            for i in range(1, dummy_period+1):
                    
                close_prices['Close (n - %s)'%i] = self.data['Close'].shift(i)
                
            self.close = close_prices.values
    
        
        
    def _get_historical_data_from_Oanda(self, instrument, size_of_history=2):
        
        params = {
          "count": size_of_history, 
          "granularity": self.time_frame,
        }
        
        
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        
        success = False
        
        while (not success):
            try:
                self.client.request(r)
                received = r.response["candles"]
                if ("c" in received[0]["mid"].keys()):
                    success = True
            except:
                success = False

        data = {
            "Open" : [],
            "High" : [],
            "Low" : [],
            "Close" : [],
            "Volume": [],
            "Date_Time" : [],        
        }

        for r in received:
            data["Open"].append(float (r["mid"]["o"]) )
            data["High"].append(float (r["mid"]["h"]))
            data["Low"].append(float (r["mid"]["l"]))
            data["Close"].append(float (r["mid"]["c"]))
            data["Volume"].append(float(r["volume"]) )
            data["Date_Time"].append(r["time"])
        
        return data
    
    def trade_with_Oanda(self):
        
        self.size_of_history = 17
        self.Oanda = True
        self._form_data(mean_std=False)

        self.last_time = self.data.index[-1].to_pydatetime()
        self.times_zones_delta = datetime.now() - self.last_time 
        
    
    def get_new_data(self):

        # wait new candel
        print ("Close time of last candle is %s" % str(self.last_time) )
        last_time = self.last_time
        while(True):
            
            while( datetime.now() - self.times_zones_delta - self.last_time  < self.delta  ):
                time.sleep(1)

            self.size_of_history = 17
            self._form_data()
            print("Generating new candle")
            if (self.last_time > last_time):
                break


    def _generate_indicators(self, data, ATR, SMA, RSI, BB):
        _high = data.High.values
        _low = data.Low.values
        _close = data.Close.values

        if ATR:
            # Compute the ATR and perform Normalisation
            data['ATR'] = talib.ATR(_high, _low, _close, timeperiod=14)
            # data['ATR'] = (data['ATR'] - np.nanmean(data['ATR']))/ np.nanstd(data['ATR'])

        if SMA:
            # Compute the SMA and perform Normalisation
            data['SMA'] = talib.SMA(_close, timeperiod=14)
            # data['SMA'] = (data['SMA'] - np.nanmean(data['SMA']))/ np.nanstd(data['SMA'])
        
        if RSI:
            data["RSI"] = talib.RSI(_close, timeperiod=14)
            # data["RSI"] = (data["RSI"] - np.nanmean(data["RSI"]))/ np.nanstd(data["RSI"])
        
        if BB:
            bands = talib.BBANDS(_close, timeperiod=14)
            data["BBhigh"] = bands[0] # (bands[0] - np.nanmean(bands[0]))/ np.nanstd(bands[0])
            data["BBlow"] = bands[1] # (bands[1] - np.nanmean(bands[1]))/ np.nanstd(bands[1])

        return data

    def _reset(self, train=True, Oanda=False):
        self.Oanda = Oanda
        
        if (self.client is None and self.Oanda):
            self.client = oandapyV20.API(access_token=self.access_token)
          
        if train:
            self.current_index = 1
            self._end = self.train_end_index
        else:
            self.current_index = self.train_end_index + 1
            self._end = self.count - 1

        self._data = self.data.iloc[self.current_index:self._end+1]

    def _step(self):
		
        if self.Oanda:
            self.current_index = -1
            self.get_new_data()
            obs = self.states[self.current_index]
            done = False
            
        else:
            obs = self.states[self.current_index]
            self.current_index += 1
            done = self.current_index > self._end
        return obs, done

class TradingEnv(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self):
        print('Please invoke .initialise_simulator() method next to complete initialization')

    
    def initialise_simulator(self, instrument, additional_pairs, accountID, access_token, input_planets_data, n_astro_periods=5, time_frame="M1",  oanda=False, size_of_history=200,  ATR=True, SMA=True, RSI=True, BB=False, trade_period=5, train_split=0.8, dummy_period=None):

        self.sim = Simulator(instrument, additional_pairs, accountID, access_token, input_planets_data=input_planets_data, n_periods=n_astro_periods, time_frame=time_frame, oanda=oanda, size_of_history=size_of_history, ATR=ATR, SMA=SMA, RSI=RSI, BB=BB, train_split=train_split, dummy_period=dummy_period)
        self.portfolio = Portfolio(prices=self.sim.data[['Open','Close']], trade_period=trade_period, train_end_index=self.sim.train_end_index)
        
        self.Oanda = oanda
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values)
    
    def _step(self, action):
    #Return the observation, done, reward from Simulator and Portfolio
        obs, done = self.sim._step()
        reward, info = self.portfolio._step(action)

        return obs, reward, done, info
    
    def _reset(self, train=True, Oanda=False):
        self.sim._reset(train=train, Oanda=Oanda)
        self.portfolio._reset(train=train, Oanda=Oanda)
    

