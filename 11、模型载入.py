from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.datasets import mnist
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from keras.src.optimizers import SGD

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
y_train = to_categorical(y_train, 10)  # 转成one-hot编码格式
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_test = to_categorical(y_test, 10)

# load_model
model = load_model('mnist.keras')
# evaluate
scores = model.evaluate(x_test, y_test)
print(scores)

# continue train
model.fit(x_train, y_train, batch_size=32, epochs=2)
scores = model.evaluate(x_test, y_test)
print(scores)


