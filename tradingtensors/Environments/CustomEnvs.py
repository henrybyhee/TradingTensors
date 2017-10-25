import time

import numpy as np
import pandas as pd

from .BaseClass import BaseEnv, BasePortfolio, BaseSimulator
from ..settings.serverconfig import GRANULARITIES, INDICATORS_SETTINGS, TF_IN_SECONDS, SYMBOL_HISTORY
from ..functions.planetry_functions import get_planet_coordinates
from ..functions.utils import OandaHandler, get_indicators, get_returns


class OandaEnv(BaseEnv):

    def __init__(self, INSTRUMENT, TIMEFRAME, train=True, _isLive=False,
                 mode='practice', additional_pairs=[], indicators=[],
                 trade_duration=1, lookback_period=0, train_split=0.7, 
                 planet_data={}, PLANET_FORWARD_PERIOD=0, PORTFOLIO_TYPE='FIXED PERIOD'):

        assert TIMEFRAME in GRANULARITIES, "Please use this timeframe format {}".format(
            GRANULARITIES)
        assert '_' in INSTRUMENT, "Please define currency pair in this format XXX_XXX"
        assert all([i in INDICATORS_SETTINGS.keys() for i in indicators]), \
        "Please use these indicators keys {}".format(INDICATORS_SETTINGS.keys())

        api_Handle = OandaHandler(TIMEFRAME, mode)
        PRECISION = api_Handle.get_instrument_precision(INSTRUMENT)

        self.sim = OandaSim(
            handle=api_Handle, TF=TIMEFRAME, SYMBOL=INSTRUMENT,
            _isLive=_isLive, other_pairs=additional_pairs,
            indicators=indicators, lookback=lookback_period,
            train_split=train_split, _isTraining=train,
            PRECISION=PRECISION,
            # Planet Data for Mr Peter's version
            planet_data=planet_data, PLANET_PERIOD=PLANET_FORWARD_PERIOD
        )

        if PORTFOLIO_TYPE == 'FIXED PERIOD':
            self.portfolio = FixedPeriodPortfolio(
                handle=api_Handle,
                DURATION=trade_duration,
                SYMBOL=INSTRUMENT,
                PRECISION=PRECISION)

        self.isTraining = train
        self.isLive = _isLive

        self.SYMBOL = INSTRUMENT

        self.api_Handle = api_Handle

        self.action_space = 3
        self.observation_space = self.sim.states_dim

    def step(self, action):
        new_obs, portfolio_feed, DONE = self.sim.step()
        ACTION, REWARD = self.portfolio.newCandleHandler(
            ACTION=action, TIME=portfolio_feed[0], 
            OPEN=portfolio_feed[1], REWARD=portfolio_feed[2])

        return new_obs, ACTION, REWARD, DONE

    def reset(self, TRAIN):
        self.isTraining = TRAIN
        states = self.sim.reset()
        self.portfolio.reset()

        return states

    '''Live Trading Functionalities
    A Threaded implementation
    '''
    def setLive(self):
        self.isLive = True
        self.lastRecordedTime = None 

    def candleListener(self, events):
        '''
        A function to detect appearance of New Candle
        Before this function, call setLive function
        '''

        self.ListenerIsAlive = True

        while True:
            
            #Get the latest Candle Time
            latestTime = self.api_Handle.getLatestTime(self.SYMBOL)

            if latestTime != self.lastRecordedTime:
                #Only happens when there is a new candle
                self.lastRecordedTime = latestTime

                print ("New Candle detected at %s"%(latestTime))

                #Put Event in the queue
                events.put("New Candle")

                #Block the queue until all texts are processed
                events.join()

                # Sleep Every 5 MINUTES
                time.sleep(300)


        
class OandaSim(BaseSimulator):

    def __init__(self, **kwargs):

        self.api_Handle = kwargs['handle']

        self.SYMBOL = kwargs['SYMBOL']

        # Attributes to create state space
        self.other_pairs = kwargs['other_pairs']  # List of other pairs
        self.indicators = kwargs['indicators']
        self.LOOKBACK = kwargs['lookback']  # how many periods to lookback
        self.planet_args = [kwargs['planet_data'], kwargs['PLANET_PERIOD']]
        

        # Attributes for training model
        # Percentage of data to be used for training, to be used in
        # define_boundaries()
        self.TRAIN_SPLIT = kwargs['train_split']
        self.isTraining = kwargs['_isTraining']  # Controlled by Environment

        # Attributes to interact with live market
        self.isLive = kwargs['_isLive']  # Flag: if True, we are trading live
        

        #For Normalization
        self.train_mean = None
        self.train_std = None

        
        self.data, self.states = self.build_data_and_states(SYMBOL_HISTORY)
        self.states_dim = self.states.shape[1]

        #To be used in every step of Simulator
        self.Open = self.data.Open.values
        self.Dates = self.data.index.to_pydatetime().tolist()

    
        #Reward: (CLOSE - OPEN) / (0.0001) 
        PRECISION = kwargs['PRECISION']
        reward_pips = (self.data['Close'] - self.data['Open']).values
        self.reward_pips = reward_pips / PRECISION 

        self.define_boundaries()
        self.reset() #Reset to initialize curr_idx and end_idx

    
    def build_data_and_states(self, HISTORY):

        # Pull primary symbol from Oanda API
        primary_data = self.api_Handle.get_history(self.SYMBOL, HISTORY)

        assert primary_data is not None, "primary_data is not DataFrame"

        states_df = pd.DataFrame(index=primary_data.index)

        states_df['Returns'] = get_returns(primary_data)

        # Compute Indicators
        if len(self.indicators) > 0:
            indie = get_indicators(primary_data, self.indicators)

            states_df = states_df.join(indie)

        #Get Return of additional pairs
        if len(self.other_pairs) > 0:

            for sym in self.other_pairs:
                _symbol_data = self.api_Handle.get_history(sym, HISTORY)
                
                assert _symbol_data is not None, "{} _symbol_data is not DataFrame"\
                .format(sym)

                _returns = get_returns(_symbol_data)
                
                #Attach to primary data
                states_df.loc[:, "%s_Returns"%sym] = _returns
        
        # Shift Data if there are any lookback period
        original = states_df.copy()
        if self.LOOKBACK > 0:
            for i in range(0,self.LOOKBACK):
                _shifted = original.shift(i+1)
                states_df = states_df.join(_shifted, rsuffix="_t-{}".format(i+1))


        # Compute Planetry data (Note: get_planet_coordinates perform shifting operation)
        if not (not self.planet_args[0]): #NOT Empty dictionary
            dates = primary_data.index.to_pydatetime().tolist()
            planet_data = get_planet_coordinates(dates, self.planet_args[0], self.planet_args[1])
            
            states_df = states_df.join(planet_data)

        states_df.dropna(axis=0, how='any', inplace=True)

        primary_data = primary_data.loc[states_df.index.tolist(), :]
        
        states = self.normalize_states(states_df.values)
        
        return primary_data, states

    '''States Normalization'''
    def normalize_states(self, states):

        if self.train_mean is None or self.train_std is None:
            self.train_mean = np.mean(states, 0)
            self.train_std = np.std(states, 0)

        transformed = (states - self.train_mean)/self.train_std    

        return transformed

    def step(self):
        
        rew = self.reward_pips[self.curr_idx] #Current Reward: Current Close - Close Open
        THIS_OPEN = self.Open[self.curr_idx] #Current Open
        THIS_TIME = self.Dates[self.curr_idx]

        self.curr_idx += 1
        done = self.curr_idx >= self.end_idx
        new_obs = self.states[self.curr_idx] #Next State 

        return new_obs, (THIS_TIME, THIS_OPEN, rew), done
    


class FixedPeriodPortfolio(BasePortfolio):
    '''
    FixedPeriodPortfolio imposes a fixed-duration trading regime
    No StopLoss is required, trades are closed automatically once they reach the specified duration
    '''
    def __init__(self, **kwargs):
        
        self.DURATION_LIMIT = kwargs['DURATION']
        self.api_Handle = kwargs['handle']
        self.SYMBOL = kwargs['SYMBOL']
        self.PRECISION = kwargs['PRECISION']
        self.reset()
        self.isLive = False

    def newCandleHandler(self, ACTION, **kwargs):
        '''
        In Live mode, step doesnt return anything
        IN Training/Testing mode, step returns action and reward
        
        TRAIN/TEST MODE:
        kwargs = {
            'TIME' : curr_time,
            'OPEN' :curr_open,
            'REWARD': reward
        }

        LIVE MODE:
        kwargs ={}
        '''

        if self.isHoldingTrade():
            #Increase trade duration
            self.curr_trade['Trade Duration'] += 1

            #Check if duration limit is reached
            reached = self.curr_trade['Trade Duration'] >= self.DURATION_LIMIT

            if reached:
                #Close Trade
                self.closeTrade(**kwargs)

            else:
                #Continue Holding

                return self.continueHolding(**kwargs)
        
        if ACTION == 2:
            # Do Nothing
            self.equity_curve.append(self.total_reward)
            REWARD = 0
            return ACTION, REWARD
        
        else:
            #TAKE A TRADE

            return self.openTrade(action=ACTION, **kwargs)


    def openTrade(self, action, **kwargs):

        if self.isLive:
            
            TYPE = 'BUY' if action == 0 else 'SELL'
            ID, TIME, PRICE = self.api_Handle.open_position(self.SYMBOL, TYPE)
            
            if ID is None or TIME is None or PRICE is None:
                #Raise Error
                print ("Failed to initiate trade")
                return
            TIME = pd.to_datetime(TIME).to_pydatetime()
            self.curr_trade['Entry Time'] = TIME
            self.curr_trade['Entry Price'] = PRICE
            self.curr_trade['ID'] = ID
            self.curr_trade['Type'] = TYPE

            self.total_trades += 1
        
            
        else:

            #Train/Test Mode
            self.total_trades += 1

            #Set cur_trade
            self.curr_trade['ID'] = self.total_trades
            TYPE = 'BUY' if action == 0 else 'SELL'
            self.curr_trade['Type'] = TYPE

            #Set Price and Time
            self.curr_trade['Entry Time'] = kwargs['TIME']
            self.curr_trade['Entry Price'] = kwargs['OPEN']

            #Manipulate reward
            rew = kwargs['REWARD']
            multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
            REWARD = rew * multiplier

           
            #Accumulate reward
            self.curr_trade['Profit'] += REWARD
            self.total_reward += REWARD

            #Update Equity 
            self.equity_curve.append(self.total_reward)

            self._isHoldingTrade = True
            return action, REWARD


    def closeTrade(self, **kwargs):
        

        if self.isLive:
            
            SYMBOL = self.curr_trade["Symbol"]
            TYPE = self.curr_trade['Type']

            #Close all position in this symbol
            closeTime, closePrice, pl =self.api_Handle.closeALLposition(SYMBOL, TYPE)

            if closeTime is None or closePrice is None or pl is None:
                #Didn't Close properly
                self.continueHolding()

            #Close successfully
            #Update curr_trade
            closeTime = pd.to_datetime(closeTime).to_pydatetime()
            self.curr_trade['Exit Time'] = closeTime
            self.curr_trade['Exit Price'] = closePrice
            self.curr_trade['Profit'] = pl/self.PRECISION

            self.journal.append(self.curr_trade)

            print (self.curr_trade)
            print ("")
            
            #reset curr_trade
            self.reset_trade()
        
        else:

            #Close the trade in Train/Test Mode
            self.curr_trade['Exit Time'] = kwargs['TIME']
            self.curr_trade['Exit Price'] = kwargs['OPEN']
            
            #Recalculate reward based on this open (More accurate) thisOpen - EntryPrice
            #Or we could leave curr_trade['Profit'] = lastClose - EntryPrice
            '''
            multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
            self.curr_trade['Profit'] = multiplier * \
            (self.curr_trade['Exit Price'] - self.curr_trade['Entry Price'])
            '''
            self.journal.append(self.curr_trade)
            self.reset_trade()

            self._isHoldingTrade = False

    def isHoldingTrade(self):

        if self.isLive:
 
            #curr_trade should have a record
            TRADE_ID = self.curr_trade["ID"]

            if TRADE_ID == 0:
                self._isHoldingTrade = False
            else:
                self._isHoldingTrade = self.api_Handle.isTradeOpen(TRADE_ID)
            
            return self._isHoldingTrade
            
        else:
            #Training/Testing 
            return self._isHoldingTrade


    def continueHolding(self, **kwargs):

        if self.isLive:
            
            
            ID = self.curr_trade['ID']
            SYMBOL = self.curr_trade['Symbol']
            pl = self.api_Handle.getOpenPL(ID, SYMBOL)
            
            self.curr_trade['Profit'] = float(pl)/self.PRECISION
            
            self.equity_curve.append(self.curr_trade['Profit'])

            return
            
        else:
            
            #Reset the action
            ACTION = 2
            
            #Manipulate reward
            rew = kwargs['REWARD']
            multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
            REWARD = rew * multiplier

            #Accumulate reward 
            self.total_reward += REWARD
            self.curr_trade['Profit'] += REWARD            
            

            #Update Equity
            self.equity_curve.append(self.total_reward)


            return ACTION, REWARD
