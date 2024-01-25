import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers


class Classifier(keras.Model):

    def __init__(self):
        super(Classifier, self).__init__()

        # unit 1, [64,64,64,1] => [32,32,32,16]
        self.conv1a = layers.Conv3D(16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv1b = layers.Conv3D(16, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max1 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 2, => [16,16,16,32]
        self.conv2a = layers.Conv3D(32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv2b = layers.Conv3D(32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max2 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 3, => [8,8,8,64]
        self.conv3a = layers.Conv3D(64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv3b = layers.Conv3D(64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max3 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 4, => [4,4,4,128]
        self.conv4a = layers.Conv3D(128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv4b = layers.Conv3D(128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max4 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 5, => [2,2,2,256]
        self.conv5a = layers.Conv3D(256, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv5b = layers.Conv3D(256, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max5 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 6, => [1,1,1,512]
        self.conv6a = layers.Conv3D(512, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv6b = layers.Conv3D(512, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.max6 = layers.MaxPool3D(pool_size=2, strides=2, padding='same')

        # unit 7, => [2]
        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)
        # self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        self.fc3 = layers.Dense(20, activation="softmax")


    def call(self, x):
        # inputs_noise: (b, 64), inputs_condition: (b, 3)
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.max1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.max2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.max3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.max4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.max5(x)

        x = self.conv6a(x)
        x = self.conv6b(x)
        x = self.max6(x)

        x = tf.keras.layers.Flatten()(x)

        x = self.fc1(x)
        x = layers.Dropout(0.5)(x)
        # x = self.fc2(x)
        label_predicted = self.fc3(x)

        return x, label_predicted


