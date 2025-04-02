from keras.src.models import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import SGD
from keras.src.datasets import mnist
from keras.src.utils import to_categorical
from keras.src.regularizers import L2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Dense(200, activation='tanh', input_dim=784, kernel_regularizer=L2(0.003)),
    Dense(100, activation='tanh', kernel_regularizer=L2(0.003)),
    Dense(10, activation='softmax', kernel_regularizer=L2(0.003))
])
sgd = SGD(learning_rate=0.2)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=30)
scores_train = model.evaluate(x_train, y_train)
scores_test = model.evaluate(x_test, y_test)
print("scores_train : ", scores_train)
print("scores_test : ", scores_test)
"""
before:
    scores_train :  [0.0024905905593186617, 1.0]
    scores_test :  [0.0710955411195755, 0.9804999828338623]

after:
    scores_train :  [0.4261242151260376, 0.9499499797821045]
    scores_test :  [0.42465293407440186, 0.9496999979019165]
"""
