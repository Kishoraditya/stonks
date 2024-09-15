from plotly.subplots import make_subplots
import plotly.graph_objects as go

def create_stock_chart(data, analysis):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Stock Price', 'RSI'), 
                        row_width=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close'],
                    name='Stock Price'), row=1, col=1)

    # Moving Average
    fig.add_trace(go.Scatter(x=data.index, y=analysis['moving_average_50'],
                    line=dict(color='orange', width=1),
                    name='50-day MA'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=analysis['upper_band'],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=analysis['lower_band'],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Lower BB'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=analysis['rsi_14'],
                    line=dict(color='purple', width=1),
                    name='RSI'), row=2, col=1)

    fig.update_layout(height=800, title_text="Stock Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig
