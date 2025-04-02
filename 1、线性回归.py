import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense

# train data
x = np.random.rand(100)
noise = np.random.normal(0, 0.01, x.shape)
y = 0.1 * x + 0.2 + noise
plt.scatter(x, y)

model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')
model.fit(x, y, batch_size=20, epochs=1000)

y_pred = model.predict(x)
plt.plot(x, y_pred)
plt.show()
