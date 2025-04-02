import numpy as np
import matplotlib.pyplot as plt
from keras.src.models import Sequential
from keras.src.layers import Dense, Activation
from keras.src.optimizers import SGD

x = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.01, x.shape)
y = np.square(x) + noise
plt.scatter(x, y)

model = Sequential([
    Dense(units=10, input_shape=(1, ), activation='tanh'),
    Dense(units=1, activation='tanh')
])
sgd = SGD(learning_rate=0.3)
model.compile(loss='mse', optimizer=sgd)
model.fit(x, y, batch_size=20, epochs=1000)

y_pred = model.predict(x)
plt.plot(x, y_pred)
plt.show()
