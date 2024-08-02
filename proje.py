
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# Veriyi yükle
df = pd.read_csv('D:/household_power_consumption.txt', sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['nan','?'])

# Eksik değerleri doldur
df = df.fillna(df.mean())

# Tarih sütununu indeks olarak ayarla
df.set_index('datetime', inplace=True)

# Günlük toplam güç tüketimini hesapla
daily_power = df.resample('D').sum()

# Sadece 'Global_active_power' sütununu al
data = daily_power['Global_active_power'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # 30 günlük veri kullanarak tahmin yapacağız
X, y = create_sequences(data_scaled, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Tahminler yap
y_pred = model.predict(X_test)

# Tahminleri orijinal ölçeğe geri dönüştür
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)


plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='Gerçek Değerler')
plt.plot(y_pred_inv, label='Tahminler')
plt.legend()
plt.title('LSTM Model Tahminleri vs Gerçek Değerler')
plt.xlabel('Gün')
plt.ylabel('Günlük Toplam Güç Tüketimi')
plt.show()
