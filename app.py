from flask import Flask, request, render_template
from priceflow import tendays
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    s_date = request.form['s_date']
    e_date = request.form['e_date']
    code = request.form['code']
    stock_info = yf.Ticker(code).info
    stock_name = stock_info['longName']    
    train_Y, test_Y, tomorrow_pred = tendays.process(s_date, e_date, code)
    return render_template('result.html', s_date=s_date, e_date=e_date, code=code, stock_name=stock_name, train_Y=train_Y, test_Y=test_Y, tomorrow_pred = tomorrow_pred)


if __name__ == '__main__':
    app.run(debug=True)