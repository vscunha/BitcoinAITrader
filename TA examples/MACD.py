import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc
from ta.trend import macd


def plot_ohcl(df, ax1_indicators=[], ax2_indicators=[]):
    df_original = df.copy()
    # necessary convert to datetime
    df['Date'] = pd.to_datetime(df.Date)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # We are using the style ‘ggplot’
    plt.style.use('ggplot')

    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16, 8))

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(
        ax1, df.values, width=0.8 / 24, colorup='green', colordown='red', alpha=0.8
    )
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # plot all ax1 indicators
    for indicator in ax1_indicators:
        ax1.plot(df['Date'], df_original[indicator], '.')

    # plot all ax2 indicators
    for indicator in ax2_indicators:
        ax2.plot(df['Date'], df_original[indicator], '-')

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()


def add_indicators(df):
    # Add Moving Average Convergence Divergence (MACD) indicator
    df['MACD'] = macd(close=df['Close'], window_slow=26, window_fast=12, fillna=True)

    return df


if __name__ == '__main__':
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = add_indicators(df)

    test_df = df[-400:]

    # Add Moving Average Convergence Divergence
    plot_ohcl(test_df, ax2_indicators=['MACD'])
