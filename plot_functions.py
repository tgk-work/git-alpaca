def plot_stock(price_data):
    # Plot (unchanged, but works with full data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    mpf.plot(price_data, type='candle', style=s, ax=ax1, ylabel='Price ($)', warn_too_much_data=2000)
    ax1.plot(price_data['EMA_20'], color='red', label='EMA20')
    ax1.plot(price_data['EMA_50'], color='orange', label='EMA50')
    ax1.plot(price_data['EMA_200'], color='green', label='EMA200')
    ax1.legend(); ax1.grid(True)

    ax2.plot(price_data['RSI'], color='purple', label='RSI')
    ax2.axhline(70, color='red', linestyle='--'); ax2.axhline(30, color='green', linestyle='--')
    ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(True)

    plt.show()