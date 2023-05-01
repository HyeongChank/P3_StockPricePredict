import yfinance as yf
import matplotlib.pyplot as plt

# 삼성전자 주가 데이터 가져오기
def load_data(s_date, e_date):

    samsung = yf.download('005930.KS', start=s_date, end=e_date)
    samsung.to_csv('samsung.csv')
    hyundai_car = yf.download('005380.KS', start=s_date, end=e_date)
    stocks = [samsung, hyundai_car]
    return stocks

# 시계열 그래프 그리기
def make_graph(stocks):
    samsung, hyundai_car = stocks    
    fig, (ax1, ax2) = plt.subplots(2,1)
    # Plot 1
    ax1.plot(samsung.index, samsung['Adj Close'], label='Samsung')
    ax1.set_title('Stock Prices')
    ax1.set_ylabel('Price')
    ax1.legend()
    # Plot 2
    ax2.plot(hyundai_car.index, hyundai_car['Adj Close'], label='Hyundai Car')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()

    # Show Plot
    plt.show()
    
if __name__=='__main__':
    s_date = '2023-04-01'
    e_date = '2023-04-30'
    stocks = load_data(s_date, e_date)
    make_graph(stocks)