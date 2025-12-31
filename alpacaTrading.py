# --------------------------------------------------------------
#  MPLD3 INTERACTIVE TRADING DASHBOARD – ULTRA-FAST
# --------------------------------------------------------------
from alpaca_keys import *
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import pytz
import time
import threading
import IPython.display as display

# ------------------------------------------------------------------
# 1. Clients & Config
# ------------------------------------------------------------------
data_client = StockHistoricalDataClient(myAPIKey, mySecretKey)
symbol = 'AAPL'
est = pytz.timezone('US/Eastern')
now_est = datetime.now(est) - timedelta(minutes=15)
N_BARS = 1000

# ------------------------------------------------------------------
# 2. Fetch data
# ------------------------------------------------------------------
def get_last_n_bars(tf, n):
    days_back = {'1Min': n/390*1.2, '5Min': n/78*1.2, '15Min': n/26*1.2}.get(str(tf), n/6.5*1.2)
    start = now_est - timedelta(days=max(int(days_back), 1))
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=now_est, limit=10000, feed='iex')
    try: bars = data_client.get_stock_bars(req).df
    except: return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex): bars = bars.xs(symbol, level=0)
    bars.index = pd.to_datetime(bars.index).tz_convert(est)
    df = bars.copy() if tf.value == 1 and tf.unit == TimeFrameUnit.Minute else bars.between_time('09:30', '16:00').copy()
    return df.iloc[-n:]

timeframes = [
    (TimeFrame(1,  TimeFrameUnit.Minute), '1m'),
    (TimeFrame(5,  TimeFrameUnit.Minute), '5m'),
    (TimeFrame(15, TimeFrameUnit.Minute), '15m'),
    (TimeFrame(1,  TimeFrameUnit.Hour),   '1h')
]

data_dict = {name: get_last_n_bars(tf, N_BARS) for tf, name in timeframes}

# ------------------------------------------------------------------
# 3. Indicators
# ------------------------------------------------------------------
def add_indicators(df):
    if df.empty: return
    close = df['close']
    for span, col in [(20, 'EMA_20'), (50, 'EMA_50'), (200, 'EMA_200')]: df[col] = close.ewm(span=span, min_periods=0).mean()
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rsi = 100 - 100/(1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan))
    df['RSI'] = 100 - 100/(1 + df['EMA_20'].diff().clip(lower=0).rolling(14).mean() / (-df['EMA_20'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))
    df['RSI_50'] = 100 - 100/(1 + df['EMA_50'].diff().clip(lower=0).rolling(14).mean() / (-df['EMA_50'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))
    df['RSI_200'] = 100 - 100/(1 + df['EMA_200'].diff().clip(lower=0).rolling(14).mean() / (-df['EMA_200'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26; df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = df['MACD'] - df['Signal']

for df in data_dict.values(): add_indicators(df)

# ------------------------------------------------------------------
# 4. MPLD3 Plot with Tooltip
# ------------------------------------------------------------------
class TooltipPlugin(plugins.PluginBase):
    JAVASCRIPT = """
    mpld3.register_plugin("tooltip", TooltipPlugin);
    TooltipPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    TooltipPlugin.prototype.constructor = TooltipPlugin;
    TooltipPlugin.prototype.draw = function(){
      var obj = mpld3.get_element(this.props.id);
      var data = this.props.data;
      obj.elements().on("mousemove", function(d, i){
        var idx = Math.round(obj.axes[0].x.invert(d3.mouse(this)[0]));
        idx = Math.max(0, Math.min(idx, data.length-1));
        var dt = data[idx][0]; var price = data[idx][1];
        d3.select(this).append("text")
          .text(dt + " | " + price.toFixed(2))
          .attr("x", d3.mouse(this)[0] + 10)
          .attr("y", d3.mouse(this)[1] - 10)
          .attr("fill", "black")
          .attr("font-size", 10)
          .attr("background", "yellow");
      }).on("mouseout", function(){ d3.select(this).selectAll("text").remove(); });
    };
    """

    def __init__(self, points, data):
        self.dict_ = {"type": "tooltip", "id": points.id, "data": data}

def create_panel(df, title):
    if df.empty: return None
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3,1,1]})
    x = np.arange(len(df))
    up = df['close'] >= df['open']; down = ~up

    # Price + Candles
    ax1.vlines(x, df['low'], df['high'], color='k', lw=0.6)
    ax1.bar(x[up], df['close'][up]-df['open'][up], bottom=df['open'][up], width=1, color='g', alpha=0.7)
    ax1.bar(x[down], df['open'][down]-df['close'][down], bottom=df['close'][down], width=1, color='r', alpha=0.7)
    ax1.plot(x, df['EMA_20'], 'red', label='EMA 20')
    ax1.plot(x, df['EMA_50'], 'orange', label='EMA 50')
    ax1.plot(x, df['EMA_200'], 'green', label='EMA 200')
    ax1.set_title(title); ax1.legend(); ax1.grid()

    # RSI
    ax2.plot(x, df['RSI'], 'red', label='RSI'); ax2.plot(x, df['RSI_50'], 'orange'); ax2.plot(x, df['RSI_200'], 'green')
    ax2.axhline(80, color='r', ls='--'); ax2.axhline(20, color='g', ls='--')
    ax2.set_ylim(-10, 110); ax2.legend(); ax2.grid()

    # MACD
    ax3.plot(x, df['MACD'], 'b'); ax3.plot(x, df['Signal'], 'r')
    ax3.bar(x, df['MACD_Hist'], color='gray', alpha=0.4, width=1)
    ax3.legend(); ax3.grid()

    # Tooltip data
    tooltip_data = [[df.index[i].strftime('%Y-%m-%d %H:%M'), df['close'].iloc[i]] for i in range(len(df))]
    points = ax1.plot(x, df['close'], 'o', alpha=0)[0]  # invisible points
    plugins.connect(fig, TooltipPlugin(points, tooltip_data))

    return fig

# ------------------------------------------------------------------
# 5. Initial Plot
# ------------------------------------------------------------------
figs = {}
for name, df in data_dict.items():
    figs[name] = create_panel(df, name.upper())

# Combine into 2x2 grid
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
for i, name in enumerate(['1m','5m','15m','1h']):
    row, col = divmod(i, 2)
    ax = fig.add_subplot(gs[row, col])
    if figs[name]:
        for line in figs[name].axes[0].lines + figs[name].axes[0].collections:
            ax.add_artist(line)
        ax.set_xlim(figs[name].axes[0].get_xlim())
        ax.set_ylim(figs[name].axes[0].get_ylim())

# ------------------------------------------------------------------
# 6. Enable mpld3
# ------------------------------------------------------------------
mpld3.enable_notebook()  # Run in Jupyter
html = mpld3.fig_to_html(fig)
display.display(display.HTML(html))

# ------------------------------------------------------------------
# 7. Live Update (Refresh HTML)
# ------------------------------------------------------------------
def live_update():
    while True:
        time.sleep(60)
        global now_est
        now_est = datetime.now(est) - timedelta(minutes=15)
        for tf, name in timeframes:
            df = get_last_n_bars(tf, N_BARS)
            if not df.empty:
                data_dict[name] = df
                add_indicators(df)
                figs[name] = create_panel(df, name.upper())
        # Rebuild combined plot
        fig.clf()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        for i, name in enumerate(['1m','5m','15m','1h']):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])
            if figs[name]:
                for line in figs[name].axes[0].lines + figs[name].axes[0].collections:
                    ax.add_artist(line)
                ax.set_xlim(figs[name].axes[0].get_xlim())
                ax.set_ylim(figs[name].axes[0].get_ylim())
        display.clear_output(wait=True)
        display.display(display.HTML(mpld3.fig_to_html(fig)))

threading.Thread(target=live_update, daemon=True).start()