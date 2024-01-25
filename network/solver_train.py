import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import time
import datetime
import argparse
import collections
import tensorflow_text as text
import tensorflow_graphics.nn.loss as cd_loss
from solver import Solver
import scipy


print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')


def plot_mul_3D_voxels(point_clouds_real, point_clouds_fake, query, save_name=None):
    print(point_clouds_real.shape)
    print(point_clouds_fake.shape)
    num_fig = point_clouds_real.shape[0]*2
    ncols = point_clouds_real.shape[0]

    fig = plt.figure(figsize=(3*ncols, 3*2))
    fig.suptitle(query, fontsize=12)
    for i in range(num_fig):
        ax = fig.add_subplot(2, ncols,i+1,projection='3d')
        if i < point_clouds_real.shape[0]:
            ax.scatter(point_clouds_real[i, :, 0], point_clouds_real[i, :, 1], point_clouds_real[i, :, 2])
        else:
            ax.scatter(point_clouds_fake[i-ncols, :, 0], point_clouds_fake[i-ncols, :, 1], point_clouds_fake[i-ncols, :, 2])
        ax.set_axis_off()
    if save_name:
        save_to = "{0}.png".format(save_name)
        plt.savefig(save_to, dpi=300)
    plt.close()


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.expand_dims(x, -1)
    y = tf.cast(y, dtype=tf.float32)
    return x, y


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/datasets", help='Dataset path')
    parser.add_argument('--dual_encoder_path', type=str, default=r"/training_results/dual_encoder", help='dual encoder path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="ratio of training dataset")
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature")
    parser.add_argument('--epochs', type=int, default=200, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--query', nargs='+', default="A disordered microstructure", metavar='N', help='text query array')
    parser.add_argument('--generation_number', type=int, default=3, help="the number of generated shape for each query")
    args = parser.parse_args()
    return args


def main():

    # set up environment
    start_time = time.time()
    args = parsing()
    create_dir("training_results/solver")

    voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(dtype=float)
    E_Ani = pd.read_csv(os.path.join(args.dataset_path, "E_Ani.csv"))
    E_label = np.expand_dims(1/E_Ani["E"], axis=-1)
    Ani_label = np.expand_dims(E_Ani["Anisotropy"], axis=-1)
    phi = np.expand_dims(np.sum(np.reshape(voxels, [2000,-1]), axis=-1)/64**3, axis=-1)
    labels = np.concatenate((phi, E_label, E_Ani), axis=-1)
    print(phi.shape)
    print(E_label.shape)

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(E_label)
    axs[1].hist(Ani_label)
    axs[2].hist(phi)
    plt.show()

    index = np.linspace(0,1999,2000).astype(dtype=int)
    index, voxels, labels = sklearn.utils.shuffle(index, voxels, labels, random_state=0)
    voxels = tf.cast(voxels, tf.float32)
    labels = tf.cast(labels, tf.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((voxels[:1600], labels[:1600]))
    train_ds = train_ds.shuffle(10000).map(preprocess).batch(32)

    val_ds = tf.data.Dataset.from_tensor_slices((voxels[1600:], labels[1600:]))
    val_ds = val_ds.map(preprocess).batch(32)

    solver = Solver()
    solver.build(input_shape=[None, 64, 64, 64, 1])
    optimizer = optimizers.Adam(learning_rate=1e-4)  ## lr = 2e-4, beta_1 = 0.9

    for epoch in range(args.epochs):
        # plot train
        for step, (x, y) in enumerate(train_ds):

            with tf.GradientTape() as tape:
                pred = solver(x)
                loss = tf.keras.metrics.mean_squared_error(y, pred)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, solver.trainable_variables)
            optimizer.apply_gradients(zip(grads, solver.trainable_variables))

        print(epoch, "loss: ", loss.numpy())

        # plot test
        total_sum = 0
        total_error = 0
        for x, y in val_ds:

            pred = solver(x)
            loss_test = tf.keras.metrics.mean_squared_error(y, pred)
            loss_test = tf.reduce_mean(loss_test)

            total_sum += 1
            total_error += loss_test

        acc = total_error / total_sum
        print(epoch, "acc: ", acc.numpy())
        if epoch == 0:
            with open('training_results/solver/accuracy.txt', 'w') as f:
                f.write("epoch " + "loss " + "acc" + "\n")
        else:
            with open('training_results/solver/accuracy.txt', 'a') as f:
                f.write(str(epoch) + " " + str(loss.numpy()) + " " + str(acc.numpy()) + "\n")

        solver.save_weights('training_results/solver/ckpt/solver_%d.ckpt' % epoch)

    with open('training_results/solver/accuracy.txt', 'a') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"Tensorflow version: " + tf.__version__ + "\n")
        f.write("Training time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()

