# 필요한 라이브러리 임포트
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# 삼성전자 주식 데이터 가져오기
samsung_stock = yf.download('005930.KS', start='2022-01-01', end='2022-04-30')

# 2023년 5월 1일 이전 데이터만 사용
data = samsung_stock.loc[:'2023-05-01']

# 종가 기준으로 데이터 선택
data = data[['Close']]

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 학습용, 검증용 데이터셋 분리
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# look_back 설정
look_back = 30

# 데이터셋 생성 함수
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back])
    return np.array(X), np.array(Y)

# 데이터셋 생성
train_X, train_Y = create_dataset(train_data, look_back)
test_X, test_Y = create_dataset(test_data, look_back)

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(train_X, train_Y, epochs=50, batch_size=32, verbose=2)

# 2023년 5월 1일까지의 데이터셋 생성
future_data = scaled_data[-look_back:]

future_X = np.array([future_data])
future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
print(future_X)
# 2023년 5월 1일의 주가 예측
prediction = model.predict(future_X)
prediction = scaler.inverse_transform(prediction)

print("2023년 5월 1일 삼성전자 주가 예측 결과: ", prediction)