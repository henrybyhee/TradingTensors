TOKEN = "" #Oanda Token
ID = "" #Oanda Account ID


#Compute Returns
# Now=t,  we only know current open and last close
# True -> (OPEN[t] - OPEN[t-1])/(OPEN[t])
# False -> (CLOSE[t-1] - CLOSE[t-2]) / CLOSE[t-2]
# Note: We won't know what today's close is until tomorrow!
RETURNS_BY_OPEN = False

#ENVIRONMENT 
SYMBOL_HISTORY = 500

#INDICATORS
from talib import ATR, SMA, RSI, BBANDS

INDICATORS_SETTINGS = {
    "ATR": {
        'requires' : ['High', 'Low', 'Close'],
        'period' : 14,
        'output_name': ['ATR'],
        'func': ATR
        },

    "SMA":{
        'requires' : ['Close'],
        'period' : 14,
        'output_name': ['SMA'],
        'func': SMA
    },

    "RSI": {
        'requires' : ['Close'],
        'period' : 14,
        'output_name': ["RSI"],
        'func': RSI
    },

    "BBANDS": {
        'requires': ['Close'],
        'period': 14,
        'output_name': ['BB_High', 'BB_Mid','BB_Low'], #in the order TA_LIB ouputs
        'func': BBANDS
    }
}



#GRANULARITY IN SECONDS
TF_IN_SECONDS = {
    'S5' :	5 ,
    'S10' :	10,
    'S15':	15,
    'S30':	30,
    'M1' :	60,
    'M2' :	120,
    'M4' :	240,
    'M5' :	300,
    'M10':	600, 
    'M15':	900, 
    'M30':	1800,
    'H1':	3600,
    'H2':	7200,
    'H3':	10800,
    'H4':	14400,
    'H6':   21600,
    'H8':	28800,
    'H12':	43200,
    'D':    86400,
    'W':	604800,
    'M':    2592000	#Assuming 30 days
}

GRANULARITIES = TF_IN_SECONDS.keys()


