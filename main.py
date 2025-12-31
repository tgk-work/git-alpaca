# --------------------------------------------------------------
#  4-PANEL TILED LAYOUT – FAST, 1-min latest, RSI -10..110
# --------------------------------------------------------------
from alpaca_keys import *
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')                     # Qt backend – faster interactivity
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import pytz
import time

# ------------------------------------------------------------------
# 1. Initialise Alpaca clients
# ------------------------------------------------------------------
trading_client = TradingClient(myAPIKey, mySecretKey, paper=True)
data_client   = StockHistoricalDataClient(myAPIKey, mySecretKey)

symbol = 'AAPL'
est    = pytz.timezone('US/Eastern')
london = pytz.timezone('Europe/London')
now_est = london.localize(datetime.now() - timedelta(minutes=15)).astimezone(est)

# ------------------------------------------------------------------
# 2. Get the **latest 1000 bars** per timeframe (IEX = free)
# ------------------------------------------------------------------
N_BARS = 1000

def get_last_n_bars(tf, n):
    if str(tf) == "1Min":
        days_back = int(n / 390 * 1.2)
    elif str(tf) == "5Min":
        days_back = int(n / 78 * 1.2)
    elif str(tf) == "15Min":
        days_back = int(n / 26 * 1.2)
    else:
        days_back = int(n / 6.5 * 1.2)

    rough_start = now_est - timedelta(days=max(days_back, 1))

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=rough_start,
        end=now_est,
        limit=10000,
        feed='iex'
    )
    try:
        bars = data_client.get_stock_bars(req).df
    except Exception as e:
        print(f"Error fetching {tf}: {e}")
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level=0)
    bars.index = pd.to_datetime(bars.index).tz_convert(est)

    if tf.value == 1 and tf.unit == TimeFrameUnit.Minute:
        df = bars.copy()
    else:
        df = bars.between_time('09:30', '16:00').copy()

    print(f"{tf} range: {df.index[0]} to {df.index[-1]}\n")
    return df.iloc[-n:]

timeframes = [
    (TimeFrame(1,  TimeFrameUnit.Minute),  '1m'),
    (TimeFrame(5,  TimeFrameUnit.Minute),  '5m'),
    (TimeFrame(15, TimeFrameUnit.Minute), '15m'),
    (TimeFrame(1,  TimeFrameUnit.Hour),    '1h')
]

data_dict = {}
for tf, name in timeframes:
    df = get_last_n_bars(tf, N_BARS)
    data_dict[name] = df

# ------------------------------------------------------------------
# 3. Compute indicators
# ------------------------------------------------------------------
def add_indicators(df):
    if df.empty: return
    df['EMA_20']  = df['close'].ewm(span=20,  min_periods=0).mean()
    df['EMA_50']  = df['close'].ewm(span=50,  min_periods=0).mean()
    df['EMA_200'] = df['close'].ewm(span=200, min_periods=0).mean()

    delta = df['EMA_20'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    df['RSI'] = 100 - 100/(1 + gain.rolling(14, min_periods=1).mean() /
                             loss.rolling(14, min_periods=1).mean().replace(0, np.nan))

    for span, name in [(50, 'RSI_50'), (200, 'RSI_200')]:
        d = df[f'EMA_{span}'].diff()
        g = d.where(d > 0, 0)
        l = -d.where(d < 0, 0)
        df[name] = 100 - 100/(1 + g.rolling(14, min_periods=1).mean() /
                                l.rolling(14, min_periods=1).mean().replace(0, np.nan))

    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

for name, df in data_dict.items():
    add_indicators(df)

# ------------------------------------------------------------------
# 4. Create tiled layout
# ------------------------------------------------------------------
fig = plt.figure(figsize=(24, 18), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05)

axs = {}
for row in range(2):
    for col in range(2):
        sub_gs = gs[row, col].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        panel_name = ['1m', '5m', '15m', '1h'][row*2 + col]
        ax_price = fig.add_subplot(sub_gs[0])
        ax_rsi   = fig.add_subplot(sub_gs[1], sharex=ax_price)
        ax_macd  = fig.add_subplot(sub_gs[2], sharex=ax_price)
        axs[panel_name] = [ax_price, ax_rsi, ax_macd]

# ------------------------------------------------------------------
# 5. Plot each timeframe
# ------------------------------------------------------------------
def plot_panel(ax_price, ax_rsi, ax_macd, df, title):
    if df.empty:
        ax_price.text(0.5, 0.5, 'No Data', transform=ax_price.transAxes,
                      ha='center', va='center', fontsize=12)
        return
    x = np.arange(len(df))
    up   = df['close'] >= df['open']
    down = ~up
    ax_price.vlines(x, df['low'], df['high'], color='black', lw=0.6)
    ax_price.bar(x[up],   df['close'][up]   - df['open'][up],
                 bottom=df['open'][up],   width=1.0, color='green', alpha=0.7)
    ax_price.bar(x[down], df['open'][down] - df['close'][down],
                 bottom=df['close'][down], width=1.0, color='red',   alpha=0.7)

    ax_price.plot(x, df['EMA_20'],  color='red',    label='EMA 20')
    ax_price.plot(x, df['EMA_50'],  color='orange', label='EMA 50')
    ax_price.plot(x, df['EMA_200'], color='green',  label='EMA 200')

    high = df['high'].max(); low = df['low'].min(); pad = (high-low)*0.05
    ax_price.set_ylim(low-pad, high+pad)
    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left', fontsize=8)
    ax_price.set_title(title)

    ax_rsi.plot(x, df['RSI'],      color='red',    label='RSI (EMA20)')
    ax_rsi.plot(x, df['RSI_50'],   color='orange', label='RSI (EMA50)')
    ax_rsi.plot(x, df['RSI_200'],  color='green',  label='RSI (EMA200)')
    ax_rsi.axhline(80, color='red',   linestyle='--')
    ax_rsi.axhline(20, color='green', linestyle='--')
    ax_rsi.set_ylim(-10, 110)
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend(loc='upper left', fontsize=8)

    ax_macd.plot(x, df['MACD'],      color='blue',  label='MACD')
    ax_macd.plot(x, df['Signal'],    color='red',   label='Signal')
    ax_macd.bar(x, df['MACD_Hist'], color='gray', alpha=0.4, width=1.0, label='Histogram')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(loc='upper left', fontsize=8)

    for ax in [ax_price, ax_rsi, ax_macd]:
        ax.grid(True, which='major', ls='-', lw=0.5, alpha=0.7)   # major grid only (faster)

plot_panel(axs['1m'][0],  axs['1m'][1],  axs['1m'][2],  data_dict['1m'],  '1-Minute')
plot_panel(axs['5m'][0],  axs['5m'][1],  axs['5m'][2],  data_dict['5m'],  '5-Minute')
plot_panel(axs['15m'][0], axs['15m'][1], axs['15m'][2], data_dict['15m'], '15-Minute')
plot_panel(axs['1h'][0],  axs['1h'][1],  axs['1h'][2],  data_dict['1h'],  '1-Hour')

# ------------------------------------------------------------------
# 6. Time labels + date change
# ------------------------------------------------------------------
def update_ticks(panel_name):
    ax_price, ax_rsi, ax_macd = axs[panel_name]
    df = data_dict[panel_name]
    if df.empty: return
    x = np.arange(len(df))
    vmin, vmax = ax_macd.get_xlim()
    tick_positions = np.linspace(vmin, vmax, 10, endpoint=True)
    tick_labels = []
    for pos in tick_positions:
        idx = int(np.clip(np.round(pos), 0, len(df)-1))
        dt = df.index[idx]
        tick_labels.append(dt.strftime('%Y-%m-%d') if dt.hour == 9 and dt.minute == 30 else dt.strftime('%H:%M'))
    for ax in [ax_price, ax_rsi, ax_macd]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    if hasattr(ax_price, 'date_artists'):
        for artist in ax_price.date_artists: artist.remove()
    ax_price.date_artists = []

    normalized = df.index.normalize()
    day_changes = df.index[normalized.diff() != pd.Timedelta(0)]
    for change in day_changes:
        idx = df.index.get_loc(change)
        if vmin <= idx <= vmax:
            line = ax_price.axvline(idx, color='black', linestyle=':', linewidth=1)
            txt = ax_price.text(idx, ax_price.get_ylim()[1], change.strftime('%Y-%m-%d'),
                                rotation=90, va='top', ha='right', fontsize=8, color='black')
            ax_price.date_artists.extend([line, txt])

for name in axs: update_ticks(name)

# ------------------------------------------------------------------
# 7. Interactive helpers – FAST, STABLE, RSI/MACD ignored
# ------------------------------------------------------------------
last_update = 0
UPDATE_INTERVAL = 0.3
panel_axes = {name: axs[name] for name in axs}
for name, panel in panel_axes.items():
    if not data_dict[name].empty:
        MultiCursor(fig.canvas, (panel[0],), color='gray', lw=1, horizOn=True, vertOn=True)

tooltips = {}
for name, panel in panel_axes.items():
    if data_dict[name].empty: continue
    ax_price = panel[0]
    tooltip = ax_price.annotate('', xy=(0,0), xytext=(10,10),
                                textcoords="offset points", visible=False)
    tooltip.set_animated(True)
    tooltips[name] = tooltip

def motion_handler(ev):
    if not ev.inaxes or ev.inaxes not in [p[0] for p in panel_axes.values()]:
        for t in tooltips.values(): t.set_visible(False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return

    panel_name = next(p for p, panel in panel_axes.items() if ev.inaxes == panel[0])
    panel = panel_axes[panel_name]

    if hasattr(ev.inaxes, '_pan_start'):
        start_x, start_y, xlim, ylim = ev.inaxes._pan_start
        dx = start_x - ev.xdata; dy = start_y - ev.ydata
        for ax in panel:
            ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        for t in tooltips.values(): t.set_visible(False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return

    df = data_dict[panel_name]
    idx = int(np.clip(round(ev.xdata), 0, len(df)-1))
    dt = df.index[idx]; price = df['close'].iloc[idx]

    tooltip = tooltips[panel_name]
    tooltip.xy = (ev.xdata, ev.ydata)
    tooltip.set_text(f'{dt:%Y-%m-%d %H:%M}\n{price:.2f}')
    tooltip.set_visible(True)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

fig.canvas.mpl_connect('motion_notify_event', motion_handler)

def on_scroll(ev):
    if not ev.inaxes or ev.inaxes not in [p[0] for p in panel_axes.values()]: return
    panel_name = next(p for p, panel in panel_axes.items() if ev.inaxes == panel[0])
    factor = 1/1.2 if ev.button == 'up' else 1.2
    x, y = ev.xdata, ev.ydata
    panel = panel_axes[panel_name]
    if ev.key == 'control':
        ylim = ev.inaxes.get_ylim(); h = ylim[1] - ylim[0]
        ev.inaxes.set_ylim(y - h*factor*(y-ylim[0])/h, y + h*factor*(ylim[1]-y)/h)
    else:
        xlim = ev.inaxes.get_xlim(); w = xlim[1] - xlim[0]
        new_left  = x - w*factor*(x-xlim[0])/w
        new_right = x + w*factor*(xlim[1]-x)/w
        for ax in panel: ax.set_xlim(new_left, new_right)
    global last_update
    now = time.time()
    if now - last_update > UPDATE_INTERVAL:
        update_ticks(panel_name); last_update = now
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

fig.canvas.mpl_connect('scroll_event', on_scroll)

def on_press(ev):
    if not ev.inaxes or ev.inaxes not in [p[0] for p in panel_axes.values()]: return
    panel_name = next(p for p, panel in panel_axes.items() if ev.inaxes == panel[0])
    panel = panel_axes[panel_name]
    for ax in panel:
        ax._pan_start = (ev.xdata, ev.ydata, ax.get_xlim(), ax.get_ylim())

fig.canvas.mpl_connect('button_press_event', on_press)

def on_release(ev):
    if not ev.inaxes or not hasattr(ev.inaxes, '_pan_start'): return
    panel_name = next(p for p, panel in panel_axes.items() if ev.inaxes == panel[0])
    update_ticks(panel_name)
    for ax in panel_axes[panel_name]:
        if hasattr(ax, '_pan_start'): del ax._pan_start
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

fig.canvas.mpl_connect('button_release_event', on_release)

fig.canvas.manager.toolbar.visible = False   # Hide toolbar

# ------------------------------------------------------------------
# 8. Show (maximized with Qt)
# ------------------------------------------------------------------
fig.canvas.manager.window.showMaximized()   # Qt method – maximizes the window
plt.show()
