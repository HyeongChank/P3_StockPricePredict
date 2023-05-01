import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# 데이터 불러오기
def load_data(s_date, e_date):
    df = pd.read_csv('samsung.csv', index_col='Date', parse_dates=True)
    df = df.loc[s_date:e_date]
    return df

# 데이터 전처리
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return scaled, scaler

# 시퀀스 데이터 생성 함수
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# LSTM 모델 생성
def create_model(look_back):
    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 예측 결과 시각화
def visualize_result(df, train_predict, test_predict, look_back):
    train_size = int(len(df) * 0.7)
    test_size = len(df) - train_size
    
    train_predict_plot = np.empty_like(df['Close'])
    train_predict_plot[:] = np.nan
    train_predict_plot[look_back:len(train_predict)+look_back] = train_predict.reshape(-1)
    
    test_predict_plot = np.empty_like(df['Close'])
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict)+(look_back*2)+1:len(df)-1] = test_predict.reshape(-1)
    
    plt.figure(figsize=(15, 5))
    plt.plot(df['Close'], label='Actual')
    plt.plot(train_predict_plot, label='Train')
    plt.plot(test_predict_plot, label='Test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 데이터 불러오기
    s_date = '2023-01-01'
    e_date = '2023-04-30'
    df = load_data(s_date, e_date)

    # 데이터 전처리
    scaled, scaler = preprocess_data(df)

    # train, test 데이터 분리
    look_back = 30
    X_train, Y_train = create_dataset(scaled, look_back)
    X_test, Y_test = create_dataset(scaled, look_back)
    
    # LSTM 모델 생성 및 학습
    model = create_model(look_back)
    model.fit(X_train, Y_train, epochs=50, batch_size=64)

    # 모델 예측
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 결과값 역스케일링
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    # 예측 결과 시각화
    visualize_result(df, train_predict, test_predict, look_back)