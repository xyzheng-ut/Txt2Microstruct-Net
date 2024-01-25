import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow_text as text
import os
import argparse
import pandas as pd
import collections
import sklearn
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from scipy import linalg
from classifier_net import Classifier

print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("tensorflow is running on GPU:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')


def read_image(image_path):
    image_path_0 = image_path+"_0.jpg"
    image_path_1 = image_path+"_1.jpg"
    image_path_2 = image_path+"_2.jpg"
    image_array = tf.concat((tf.image.decode_jpeg(tf.io.read_file(image_path_0), channels=1),
                             tf.image.decode_jpeg(tf.io.read_file(image_path_1), channels=1),
                             tf.image.decode_jpeg(tf.io.read_file(image_path_2), channels=1)),
                            axis=-1)
    return tf.image.resize(image_array, (299, 299))


def find_matches(image_embeddings, query_embedding, k=10, normalize=True):
    # Normalize the query and the image embeddings.
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    # Retrieve top k indices.
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    # Return matching image paths.
    return results[0]


def plot_mul_3D_voxels(voxels, query, save_name=None):

    # voxels = tf.concat((voxels_real, voxels_fake), 0)
    # voxels = (voxels+1)/2
    voxels = np.squeeze(np.round(voxels).astype(dtype=bool))

    # voxels = np.round(voxels).astype(dtype=bool)
    # voxels = np.squeeze(voxels)
    num_fig = voxels.shape[0]
    ncols = voxels.shape[0]

    fig = plt.figure(figsize=(3*ncols, 3*1))
    fig.suptitle(query, fontsize=12)
    for i in range(num_fig):
        ax = fig.add_subplot(1, ncols,i+1,projection='3d')
        ax.voxels(voxels[i], facecolors="C2", edgecolor=None)
    if save_name:
        save_to = "{0}.png".format(save_name)
        plt.savefig(save_to, dpi=300)
    # plt.close()


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(2048, activation="relu")
        self.fc3 = layers.Dense(4096, activation="relu")
        self.fc4 = layers.Dense(4096, activation="relu")
        self.noise = layers.GaussianNoise(0.1)
        self.z_mean = layers.Dense(2560, name="z_mean")  # latent dim = 128
        self.z_log_var = layers.Dense(2560, name="z_log_var")  # latent dim = 128
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, img_embedding, training=None):
        x = tf.cast(img_embedding, dtype=tf.float32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # x = self.noise(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])

        return [z_mean, z_log_var, z]

    def train_step(self, data):
        # Unpack the data.
        image_embedding, x_mean_x_log_var = data
        x_mean = x_mean_x_log_var[:,:,:256]
        x_log_var = x_mean_x_log_var[:,:,256:]
        x_mean = tf.reshape(x_mean, shape=[tf.shape(x_mean)[0], -1])
        x_log_var = tf.reshape(x_log_var, shape=[tf.shape(x_log_var)[0], -1])
        # Train the discriminator.
        with tf.GradientTape() as tape:
            [z_mean, z_log_var, _] = self(image_embedding)
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(x_mean, z_mean), axis=0
                ) +
                tf.reduce_sum(
                    keras.losses.mse(x_log_var, z_log_var), axis=0
                )
            )
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result()
        }


def calculate_fid(model, fake_data, real_data):
    print(fake_data.shape)
    print(real_data.shape)
    # calculate activations
    fake_data = tf.data.Dataset.from_tensor_slices(fake_data)
    fake_data = fake_data.batch(8)
    for step, x in enumerate(fake_data):
        embedding, _ = model(x)
        if step==0:
            act1 = embedding
        else:
            act1 = tf.concat((act1, embedding), axis=0)
    act1 = np.reshape(np.array(act1),[-1, 1024])

    real_data = tf.data.Dataset.from_tensor_slices(real_data)
    real_data = real_data.batch(8)
    for step, x in enumerate(real_data):
        embedding, _ = model(x)
        if step == 0:
            act2 = embedding
        else:
            act2 = tf.concat((act2, embedding), axis=0)
    act2 = np.reshape(np.array(act2), [-1, 1024])

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/datasets", help='Dataset path')
    parser.add_argument('--dual_encoder_path', type=str, default=r"/training_results/dual_encoder", help='dual encoder path')
    parser.add_argument('--vae_path', type=str, default=r"/training_results/vae", help='vae path')
    parser.add_argument('--mlp_path', type=str, default=r"/training_results/mlp", help='mlp path')
    parser.add_argument('--classifer_ckpt', type=str, default=r"/training_results/classifier/ckpt/classifier_100.ckpt", help='classifier path')
    parser.add_argument('--query', nargs='+', default="Randomly shaped polyhedral voids within the closed-cell foam network.", metavar='N', help='text query array')
    parser.add_argument('--number', type=int, default=3, help="the number of generated shape for each query")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    args = parser.parse_args()
    return args


def main():
    args = parsing()
    # create_dir("training_results/inference")

    print("Loading vision and text encoders...")
    text_encoder = keras.models.load_model(os.path.join(args.dual_encoder_path, "text_encoder"))
    print("Models are loaded.")

    print("Loading VAE...")
    decoder = keras.models.load_model(os.path.join(args.vae_path, "decoder"))
    print("VAE model are loaded.")

    print("Loading MLP...")
    mlp = keras.models.load_model(args.mlp_path)
    print("MLP model is loaded.")
    classifier = Classifier()
    classifier.load_weights(args.classifer_ckpt)
    voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(dtype=float)
    #generate shape from query
    annotation_file = os.path.join(args.dataset_path, "captions.csv")
    annotations = pd.read_csv(annotation_file)["Captions"]
    labels = np.zeros(shape=[2000,20])
    for i in range(20):
        for j in range(100):
            labels[i*100+j, i] = 1
    index = np.linspace(0,1999,2000).astype(dtype=int)
    index = sklearn.utils.shuffle(index, random_state=0)
    voxels = tf.cast(voxels, dtype=tf.float32)
    voxels = tf.expand_dims(voxels, -1)  # [2000, 64, 64, 64, 1]

    total_sum = 0
    total_error = 0
    for i in index[:200]:
        query_embedding = text_encoder(tf.convert_to_tensor([annotations[i]]))
        [y_mean, y_log_var, _] = mlp(query_embedding)
        y_mean = tf.reshape(y_mean, shape=[-1, 256])
        y_log_var = tf.reshape(y_log_var, shape=[-1, 256])
        y_mean = y_mean * 2.5
        y_log_var = y_log_var*-10
        y = Sampling()([y_mean, y_log_var])
        generated_voxels = decoder(y)
        # plot_mul_3D_voxels(generated_voxels[:args.number], query=annotations[i])
        # plt.show()
        # generated_voxels = tf.round(generated_voxels)

        if i == index[0]:
            generated_voxels_total = generated_voxels
        else:
            generated_voxels_total = tf.concat((generated_voxels_total, generated_voxels), axis=0)

        # calculate accuracy
        generated_voxels = tf.round(generated_voxels)
        # real = tf.expand_dims(voxels[i], axis=0)
        _, pred = classifier(generated_voxels)
        y = tf.tile([labels[i]], [10,1])
        loss_test = tf.keras.metrics.categorical_accuracy(y, pred)
        loss_test = tf.reduce_mean(loss_test)
        total_sum += 1
        total_error += loss_test

    acc = total_error / total_sum
    print("acc: ", acc.numpy())  # 0.8695002

    fid = calculate_fid(classifier, generated_voxels_total, voxels)
    print(fid)  # 72.07539082015413


if __name__ == '__main__':
    main()