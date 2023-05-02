from flask import Flask, request, render_template
from priceflow import tendays
from priceflow import look_backN
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
    new_look_back = request.form['new_look_back']
    stock_info = yf.Ticker(code).info
    stock_name = stock_info['longName']    
    train_Y, test_Y, tomorrow_pred = tendays.process(s_date, e_date, code, new_look_back)
    return render_template('result.html', s_date=s_date, e_date=e_date, code=code, stock_name=stock_name, train_Y=train_Y, test_Y=test_Y, tomorrow_pred = tomorrow_pred)

@app.route('/test')
def test():

    return render_template('test.html')
@app.route('/testresult', methods=['POST'])
def testresult():
    new_look_back = request.form['new_look_back']
    s_date = '2022-03-01'
    e_date = '2023-04-30'
    # 코스닥 종목은 .KQ / 코스피 종목은 .KS
    code = '005930.KS'    
    lookback = look_backN.process(s_date, e_date, code, new_look_back)
    return render_template('testresult.html')

if __name__ == '__main__':
    app.run(debug=True)