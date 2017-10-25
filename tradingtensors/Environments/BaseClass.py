
from ..functions.utils import OandaHandler
import logging
from abc import abstractmethod

class BaseEnv(object):

    @abstractmethod
    def __init__(self):
        self.action_space = 3

    @abstractmethod
    def step(self, action):
        pass
    @abstractmethod
    def reset(self):
        pass

    @property
    def isTraining(self):
        return self._isTraining

    @isTraining.setter
    def isTraining(self, value):
        self.sim.isTraining = value
        self._isTraining = value

    @property
    def isLive(self):
        return self._isLive

    @isLive.setter
    def isLive(self, value):
        self._isLive = value
        self.sim.isLive = value
        self.portfolio.isLive = value

class BaseSimulator(object):
    
    '''
    Must Define:
    self.data: Complete Dataframe of entire state space
    self.states: numpy matrix form of self.data
    
    '''
    @abstractmethod
    def __init__(self):
        self.states = None
        self.data =None
        self.TRAIN_SPLIT = 0
        
    '''
    Reset Function:
    set cur_idx to the first instance
    set end_idx to the last instance

    Return the first instance
    '''
    def reset(self):
        
        if self.isTraining:
            self.curr_idx = self.train_start_idx
            self.end_idx = self.train_end_idx
        else:
            self.curr_idx = self.test_start_idx
            self.end_idx = self.test_end_idx

        '''Edge Case: Step function will cross boundary of data '''
        if self.curr_idx == self.end_idx:
            raise Exception("Please use more history!")

        #Return the first instance of the state space
        return self.states[self.curr_idx]


    '''
    Define the first and last index of states during training
    '''
    def define_boundaries(self):
        
        '''Before executing, check that states is defined'''
        assert self.states is not None, "No state space!"

        data_count = self.states.shape[0] 

        '''Define boundary index for training and testing'''
        self.train_start_idx = 0
        self.train_end_idx = int(self.TRAIN_SPLIT*data_count)
        self.test_start_idx = self.train_end_idx +1 
        self.test_end_idx = data_count - 1

'''
Class that keeps track of trade details,
and most importantly gives reward to agent
'''
class BasePortfolio(object):

    @abstractmethod
    def step(self, action, prices):
        pass


    def reset_trade(self):
        
        self.curr_trade = {
            'ID':0,
            'Entry Price':0, 
            'Exit Price':0, 
            'Entry Time':None, 
            'Exit Time':None ,
            'Profit':0, 
            'Trade Duration':0, 
            'Type':None,
            'Symbol': self.SYMBOL
            }

    def reset(self):
        
        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        self.average_profit_per_trade = 0

        #History of cumulative reward 
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT
        
        #Trade Profile
        self.curr_trade = {
            'ID': 0,
            'Entry Price':0, 
            'Exit Price':0, 
            'Entry Time':None, 
            'Exit Time':None ,
            'Profit':0, 
            'Trade Duration':0, 
            'Type':None,
            'Symbol': self.SYMBOL
            }

        self.journal = [] #Collection of trades

        #Flag to check if there are any existing positions
        self._isHoldingTrade = False






