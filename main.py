#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/03/30 22:17:31
@Author  :   Zhong-master 
@Version :   1.0
@Contact :   736428948@qq.com
'''

import os
""" 我电脑上装的是CPU版的，会有一个提示:
This TensorFlow binary is optimized with oneAPI Deep Neural Network 
Library (oneDNN) to use the following CPU instructions in performance-critical 
operations:  AVX AVX2
因此通过下面这条语句屏蔽这个提示 """
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Dropout
from keras.backend import abs, square, maximum, mean, log, sqrt, sum, binary_crossentropy
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical, Sequence
from sklearn.manifold import TSNE

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # 这一句只是为了解决我电脑上matplotlib.pyplot.show()不显示图片的问题


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-batch", type=int, default=32, help="Batch_size")
    parser.add_argument("-epochs", type=int, default=10, help="Epochs")
    parser.add_argument("-TSNE", type=str, default="./file/TSNE可视化.png", help="TSNE result save path")
    args = parser.parse_args()
    print("------------------args------------------")
    for k in list(vars(args).keys()):
        print("%s: %s" % (k, vars(args)[k]))
    print("----------------------------------------")
    return args

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, num_classes):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = num_classes
    
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_1, batch_x_2, batch_y = [], [], []
        while len(batch_x_1) < self.batch_size:
            idx_1, idx_2 = np.random.randint(0, len(self.x)), np.random.randint(0, len(self.x))
            if self.y[idx_1] < self.y[idx_2]:
                batch_x_1.append(self.x[idx_1])
                batch_x_2.append(self.x[idx_2])
                batch_y.append(1)
        while len(batch_x_1) < self.batch_size:
            idx_1, idx_2 = np.random.randint(0, len(self.x)), np.random.randint(0, len(self.x))
            if self.y[idx_1] != self.y[idx_2]:
                batch_x_1.append(self.x[idx_1])
                batch_x_2.append(self.x[idx_2])
                batch_y.append(0)

        batch_x_1 = np.array(batch_x_1).astype('float32') / 255.0
        batch_x_2 = np.array(batch_x_2).astype('float32') / 255.0
        batch_y = to_categorical(batch_y, num_classes=2)
        return [batch_x_1, batch_x_2], batch_y


def contrastive_loss(y_true, y_pred, margin=1):
    cross_entropy = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    loss = (1 - y_true) * cross_entropy + y_true * square(maximum(margin - y_pred, 0))
    return loss


def TSNE_SHOW(Model, arg, inputs, encoded, x_train):
    encoder = Model(inputs=inputs, outputs=encoded)
    x_train_encoded = encoder.predict(x_train.astype('float32') / 255.0)
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=0)
    x_train_tsne = tsne.fit_transform(x_train_encoded)

    plt.figure(figsize=(10, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(10):
        plt.scatter(x_train_tsne[y_train == i, 0], x_train_tsne[y_train == i, 1], c=colors[i], label=str(i))
    plt.legend()
    plt.savefig(arg.TSNE)


def MLP(Model, arg, inputs, encoded, y_test, x_test):
    encoder = Model(inputs=inputs, outputs=encoded)
    x_test_encoded = encoder.predict(x_test.astype('float32') / 255.0)
    mlp_model = Sequential([
        Dense(128, activation='relu', input_shape=(128, )),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    mlp_model.compile(optimizer=SGD(learning_rate=arg.learning_rate), loss=contrastive_loss, metrics=["accuracy"])
    history = mlp_model.fit(x_test_encoded, to_categorical(y_test), epochs=arg.epochs, batch_size=arg.batch, validation_split=0.1)
    test_loss, test_acc = mlp_model.evaluate(encoder.predict(x_test), to_categorical(y_test))
    print('Test accuracy:', test_acc)


if __name__ == "__main__":
    arg = args()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_generator = DataGenerator(x_train, y_train, batch_size=arg.batch, num_classes=2)

    shape = (28, 28, 1)
    inputs_1 = Input(shape=shape)
    inputs_2 = Input(shape=shape)
    shared_layers = Sequential([
        Conv2D(28, (3, 3), activation='relu', input_shape=shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='sigmoid'),
    ])

    encoded_1 = shared_layers(inputs_1)
    encoded_2 = shared_layers(inputs_2)
    distance = Lambda(lambda x: abs(x[0] - x[1]))([encoded_1, encoded_2])
    outputs = Dense(units=2, activation='softmax')(distance)
    model = Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model.compile(optimizer=SGD(learning_rate=arg.learning_rate), loss=contrastive_loss)
    model.fit(train_generator, epochs=arg.epochs)

    TSNE_SHOW(Model=Model, arg=arg, inputs=inputs_1, encoded=encoded_1, x_train=x_train)
    MLP(Model=Model, arg=arg, inputs=inputs_1, encoded=encoded_1, y_test=y_test, x_test=x_test)

