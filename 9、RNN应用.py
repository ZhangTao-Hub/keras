from keras.src.datasets import mnist
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    LSTM(units=50, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)
scores = model.evaluate(x_test, y_test)
print(scores)