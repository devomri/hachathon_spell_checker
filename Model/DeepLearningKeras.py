import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import DataAccess.DataAccess as data_access
import string


dal = data_access.DataAccess()

def convert_data(data):
    x = list()
    y = list()

    for z in data:
        x.append(convert_word_to_vec(z.x))
        y.append(convert_label_to_vec(z.y))

    return {
        "x": np.array(x),
        "y": np.array(y)
    }


def convert_word_to_vec(word):
    vec = np.zeros(26)
    for c in word.lower():
        i = string.ascii_lowercase.index(c)
        vec[i] += 1

    return vec


def convert_label_to_vec(label):
    vec = np.zeros(5000)
    i = dal.dictionary.index(label)
    vec[i] = 1

    return vec

batch_size = 32
epochs = 5

print('Loading data...')
train_raw = convert_data(dal.train_data)
test_raw = convert_data(dal.test_data)

x_train = train_raw["x"]
y_train = train_raw["y"]
x_test = test_raw["x"]
y_test = test_raw["y"]

num_classes = 5000

#tokenizer = Tokenizer(num_words=max_words)
#x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
#x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(26,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
#model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])