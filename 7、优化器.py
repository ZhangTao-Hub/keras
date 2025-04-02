from keras.src.datasets import mnist
from keras.src.models import Sequential
from keras.src.utils import to_categorical
from keras.src.optimizers import Adam
from keras.src.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Dense(100, input_dim=784, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20)
scores_train = model.evaluate(x_test, y_test)
scores_test = model.evaluate(x_test, y_test)
print(scores_train)
print(scores_test)
