from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from keras.src.optimizers import SGD

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
y_train = to_categorical(y_train, 10)  # 转成one-hot编码格式
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_test = to_categorical(y_test, 10)

model = Sequential([
    Dense(100, activation='tanh', input_dim=784),
    Dense(10, activation='softmax'),
])

model.compile(optimizer=SGD(learning_rate=0.2), loss="mse", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=8)

loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)
model.save('mnist.keras')

