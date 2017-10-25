import matplotlib.dates as mdates
import matplotlib.finance as mf
import matplotlib.pyplot as plt
import numpy as np

CHART_SIZE = (20,10)

def rewardPlot(record, best_models, TYPE, top_n=3):

    arr = np.asarray(record)
    
    #Return the index based on top n
    top_10_episodes = [x[0] for x in best_models]
    top_10_index = np.array(top_10_episodes) -1

    top_n_episodes = top_10_episodes[:top_n]
    top_n_index = top_10_index[:top_n]
    top_n_rewards = arr[top_n_index]


    fig = plt.figure(figsize=CHART_SIZE)
    ax = fig.add_subplot(111)
    color = 'b-' if TYPE=='Total' else 'r-'
    ax.plot(record, color)
    ax.set_title("%s Reward (Showing Top %s)"%(TYPE,top_n), fontdict={'fontsize':20})
    ax.set_xlabel("Episodes")
    

    textString = "TOP {}: \n".format(top_n)
    for i, r in enumerate(top_n_rewards):
        
        epi= top_n_episodes[i]

        textString += "Episode {}: {} \n".format(epi, record[epi-1])
    
    ax.text(0.75, 0.5, textString, fontsize=10, verticalalignment='top',transform=ax.transAxes,
    bbox={'alpha':0.5, 'pad':10})

    plt.show()


def ohlcPlot(journal, ohlc, equity_curve, PRECISION=0.0001):

    #Filter out buys and sells
    buys = [x for x in journal if x['Type']=='BUY']
    sells = [x for x in journal if x['Type']=='SELL']

    #make OHLC ohlc matplotlib friendly
    datetime_index = mdates.date2num(ohlc.index.to_pydatetime())
    
    proper_feed = list(zip(
        datetime_index, 
        ohlc.Open.tolist(), 
        ohlc.High.tolist(), 
        ohlc.Low.tolist(), 
        ohlc.Close.tolist()
        ))

    #actual PLotting
    fig, (ax, ax2) = plt.subplots(2,1, figsize=CHART_SIZE)

    ax.set_title('Action History', fontdict={'fontsize':20})
    
    all_days= mdates.DayLocator()
    ax.xaxis.set_major_locator(all_days)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    #Candlestick chart
    mf.candlestick_ohlc(
        ax,
        proper_feed,
        width=0.02,
        colorup='green',
        colordown='red'
    )


    #Buy indicator
 
    ax.plot(
        mdates.date2num([buy['Entry Time'] for buy in buys]),
        [buy['Entry Price']-0.001 for buy in buys],
        'b^',
        alpha=1.0
    )

    #Sell indicator
    ax.plot(
        mdates.date2num([sell['Entry Time'] for sell in sells]),
        [sell['Entry Price']+0.001 for sell in sells],
        'rv',
        alpha=1.0
    )


    #Secondary Plot
    ax2.set_title("Equity")
    ax2.plot(equity_curve)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.show()

