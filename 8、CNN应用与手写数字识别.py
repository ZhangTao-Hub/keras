from keras.src.datasets import mnist
from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.src.utils import to_categorical
from keras.src.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.
x_test = x_test.reshape(-1, 28, 28, 1) / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2, strides=2, padding='same'),
    Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2, strides=2, padding='same'),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

adam = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)
scores = model.evaluate(x_test, y_test)
print(scores)
"""
[0.02129390276968479, 0.9919000267982483]
"""

