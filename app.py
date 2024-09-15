from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
from analysis import analyze_stock
from visualization import create_stock_chart
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock', methods=['POST'])
def stock():
    symbol = request.form['symbol']
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")  
    
    analysis = analyze_stock(data)
    chart = create_stock_chart(data, analysis)
    chart_html = pio.to_html(chart, full_html=False)
    
    return render_template('stock.html', symbol=symbol, data=data, chart=chart_html, analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True)
