# TradingTensors 

A simple trading simulator inspired by the OpenAI framework. The objective of the project is to explore the viability of AI (Supervised/ Reinforcement) algorithms in the financial markets. Models built in Tensorflow. Designed to carry out historical backtesting and live trading with Oanda API. 



## Experiments
- [Base Environment Representation](https://github.com/Henry-bee/TradingTensors/blob/master/Examples/Base%20Environment.ipynb)
- [Experiment #1: Trading with previous n periods of Open-Open Return (DQN)](https://github.com/Henry-bee/TradingTensors/blob/master/Examples/Experiment%201%20Basic%20DQN%20Model%20and%20walkthrough.ipynb)
- [Experiment #2: Trading with Moon Coordinates? (DQN)](https://github.com/Henry-bee/TradingTensors/blob/master/Examples/Experiment%202%20DQN%20with%20planetry%20data%20.ipynb)

Sources:
[DQN made simple by Arthur Juliani](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)


## Prerequisites
- numpy
- tensorflow
- talib
- pandas
- [swisseph](https://github.com/astrorigin/pyswisseph) (Swiss Ephemeris)

## Getting Started

1.Installation:
```
pip install -e .
```

2. Enter Oanda API ID and token key in serverconfig.py

3. Best viewed in notebooks



This is by Henry Bee, intended for Mr Peter's use
