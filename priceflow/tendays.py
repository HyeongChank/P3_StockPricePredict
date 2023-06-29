import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import os
import datetime


def load_data_scale(code, s_date, e_date):
    stock = yf.download(code, start=s_date, end=e_date)
    print(stock)
    print(stock.shape)
    stock = stock[['Close']]
    stock = stock.dropna()
    # 데이터 정규화 (0 ~ 1 사이의 값으로 스케일링)
    scaler = MinMaxScaler()
    stock = scaler.fit_transform(stock)
    # 학습용 데이터와 검증용 데이터 분리 (train:test = 7:3)
    train_size = int(len(stock) * 0.7)
    test_size = len(stock) - train_size
    train, test = stock[0:train_size,:], stock[train_size:len(stock),:]
    return scaler, train, test

# LSTM 입력 데이터 생성 함수
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def make_model(train, test, ref_back, epoch, scaler):
    look_back = ref_back
    train_X, train_Y = create_dataset(train, look_back)
    test_X, test_Y = create_dataset(test, look_back)

    # 입력 데이터의 shape 변환 (samples, time steps, features)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # LSTM 모델 구성
    model = Sequential()
    model.add(LSTM(128, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 모델 학습
    model.fit(train_X, train_Y, epochs=epoch, batch_size=32, verbose=2)
    # 예측 결과
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    # 내일의 주가 예측
    last_ref_data = train_X[-1, :, :]
    last_ref_data = np.reshape(last_ref_data, (1, ref_back, 1))
    tomorrow_pred = model.predict(last_ref_data)
    tomorrow_pred = scaler.inverse_transform(tomorrow_pred)
    print("Tomorrow's predicted price: ", tomorrow_pred)

    return train_predict, test_predict, train_Y, test_Y, tomorrow_pred

def make_unscale(train_predict, test_predict, train_Y, test_Y, scaler):
    # 정규화 된 값을 다시 원래의 값으로 변환
    train_predict = scaler.inverse_transform(train_predict)
    train_Y = scaler.inverse_transform([train_Y])
    test_predict = scaler.inverse_transform(test_predict)
    test_Y = scaler.inverse_transform([test_Y])
    # 예측값
    print(train_Y)
    # 실제값
    print(test_Y)
    return train_predict, train_Y, test_predict, test_Y

def predict_to_df(train_predict, train_Y, test_predict, test_Y):
    # 예측 결과를 데이터프레임에 저장
    train_predict_df = pd.DataFrame(train_predict, columns=['train_predict'])
    train_Y_df = pd.DataFrame(train_Y[0], columns=['train_actual'])
    test_predict_df = pd.DataFrame(test_predict, columns=['test_predict'])
    test_Y_df = pd.DataFrame(test_Y[0], columns=['test_actual'])
    return train_predict_df, train_Y_df

def make_graph(train_predict_df, train_Y_df, code, s_date, e_date):
    stock_rd = yf.download(code, start=s_date, end=e_date)
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=1.5)
    # Plot 1
    ax1.plot(stock_rd.index, stock_rd['Adj Close'], label='stock')

    s_date = datetime.datetime.strptime(s_date, '%Y-%m-%d')
    e_date = datetime.datetime.strptime(e_date, '%Y-%m-%d')
    ax1.set_title('Stock Prices ({0} - {1})'.format(s_date, e_date))

    ax1.set_ylabel('Price')
    ax1.tick_params(axis='x', labelsize=6)   
    ax1.legend()
    # Plot 2
    ax2.plot(train_predict_df, label='train predict')
    ax2.plot(train_Y_df, label='train actual')
    ax2.set_title('Predict Prices')
    ax2.set_ylabel('Price')
    ax2.legend()
    # 저장 경로 설정
    save_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'graph.png')
    # 그래프 저장
    plt.savefig(save_path)
    return save_path
    # plt.show()

def process(s_date, e_date, code, new_look_back):
    epoch = 100
    # ref_back(start 4월 1일 end 4월 30일 ref_bake 10 이면 3월 20일~ 4월 20일 까지 데이터 이용)
    ref_back = int(new_look_back)
    print(type(ref_back))
    scaler, train, test = load_data_scale(code, s_date, e_date)
    train_predict, test_predict, train_Y, test_Y, tomorrow_pred = make_model(train, test, ref_back, epoch, scaler)

    train_predict, train_Y, test_predict, test_Y = make_unscale(train_predict, test_predict, train_Y, test_Y, scaler)
    train_predict_df, train_Y_df = predict_to_df(train_predict, train_Y, test_predict, test_Y)

    make_graph(train_predict_df, train_Y_df, code, s_date, e_date)
    return train_Y, test_Y, tomorrow_pred

# if __name__=='__main__':
#     s_date = '2023-03-01'
#     e_date = '2023-04-30'
#     # 코스닥 종목은 .KQ / 코스피 종목은 .KS
#     code = '005930.KS'
#     process(s_date, e_date, code)
    
    

  