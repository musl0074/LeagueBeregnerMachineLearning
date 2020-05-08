from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = np.array([[2, 7, 4, 8546, 1, 7, 2, 6.7, 0,0,1,0],
                    [2, 6, 8, 7229, 2, 1, 4, 5.7, 0, 0, 1, 0],
                    [2, 6, 5, 10147, 0, 5, 1, 5.7, 0, 0, 0, 1],
                    [9, 6, 3, 14306, 0, 0, 1, 7, 0, 0, 1, 0],
                    [3, 5, 10, 6784, 3, 15, 1, 1.4, 0, 0, 1, 0],
                    [3, 2, 10, 5175, 1, 8, 1, 5.8, 0, 0, 1, 0],
                    [9, 3, 9, 19556, 0, 1, 0, 6.7, 0, 1, 0, 0],
                    [7, 4, 4, 14821, 3, 9, 1, 7.6, 0, 0, 1, 0],
                    [11, 4, 5, 20904, 0, 2, 1, 8.6, 0, 1, 0, 0],
                    [0, 5, 14, 8221, 5, 20, 2, 0.8, 0, 0, 1, 0],
                    [2, 7, 4, 15747, 1, 10, 1, 6.8, 0, 0, 1, 0],
                    [9, 1, 12, 22201, 2, 2, 5, 6.8, 1, 0, 0, 0],
                    [13, 3, 3, 18488, 1, 9, 0, 6.6, 0, 0, 1, 0],
                    [8, 7, 13, 25573, 2, 12, 3, 7.7, 0, 1, 0, 0],
                    [5, 8, 14, 11932, 2, 23, 5, 0.6, 0, 0, 1, 0],
                    [1, 6, 4, 14068, 0, 9, 0, 5.9, 0, 0, 0, 1],
                    [11, 8, 5, 19464, 0, 3, 0, 4.4, 0, 0, 0, 1],
                    [2, 9, 5, 12716, 0, 8, 1, 7.4, 0, 0, 0, 1],
                    [5, 7, 11, 24793, 2, 11, 3, 7.5, 0, 0, 1, 0],
                    [7, 7, 5, 11014, 6, 23, 14, 1.5, 0, 1, 0, 0],
                    [23, 4, 14, 42012, 2, 2, 4, 6.6, 1, 0, 0, 0],
                    [8, 2, 5, 21200, 1, 7, 2, 9.1, 1, 0, 0, 0],
                    [10, 7, 8, 36737, 8, 14, 2, 6.4, 1, 0, 0, 0]])

X = dataset[:, 0:8]
y = dataset[:, 8:13]

#Creating model and Dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax')) #Sigmoid insted of relu for final prediction

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X, y, epochs=2000)

scores=model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))