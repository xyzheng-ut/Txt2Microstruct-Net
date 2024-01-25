import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import time
import datetime
import argparse
import collections
import tensorflow_text as text
print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')


def plot_mul_3D_voxels(voxels_real, voxels_fake, query, save_name=None):

    voxels = tf.concat((voxels_real, voxels_fake), 0)
    # voxels = (voxels+1)/2
    voxels = np.squeeze(np.round(voxels).astype(dtype=bool))

    # voxels = np.round(voxels).astype(dtype=bool)
    # voxels = np.squeeze(voxels)
    num_fig = voxels.shape[0]
    ncols = voxels.shape[0]//2

    fig = plt.figure(figsize=(3*ncols, 3*2))
    fig.suptitle(query, fontsize=12)
    for i in range(num_fig):
        ax = fig.add_subplot(2, ncols,i+1,projection='3d')
        ax.voxels(voxels[i], facecolors="C2", edgecolor=None)
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


def find_matches(image_embeddings, query_embedding, k=3, normalize=True):
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


def read_image(image_path):
    image_path_0 = image_path+"_0.jpg"
    image_path_1 = image_path+"_1.jpg"
    image_path_2 = image_path+"_2.jpg"
    image_array = tf.concat((tf.image.decode_jpeg(tf.io.read_file(image_path_0), channels=1),
                             tf.image.decode_jpeg(tf.io.read_file(image_path_1), channels=1),
                             tf.image.decode_jpeg(tf.io.read_file(image_path_2), channels=1)),
                            axis=-1)
    return tf.image.resize(image_array, (299, 299))


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Decoder(keras.Model):

    def __init__(self, ks=3):
        super(Decoder, self).__init__()
        self.fc = layers.Dense(8 * 8 * 8 * 512)
        self.bn0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv1 = layers.Conv3DTranspose(256, kernel_size=ks, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = layers.Conv3DTranspose(128, kernel_size=ks, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = layers.Conv3DTranspose(64, kernel_size=ks, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv4 = layers.Conv3DTranspose(32, kernel_size=ks, strides=1, padding='same')
        self.bn4 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv6 = layers.Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same')

    def call(self, voxel_embedding, training=None):
        # inputs_noise: (b, 128)
        voxel_embedding = tf.cast(voxel_embedding, dtype=tf.float32)
        net = self.fc(voxel_embedding)
        net = self.bn0(net, training=training)
        net = tf.reshape(net, [-1, 8, 8, 8, 512])
        net = tf.nn.leaky_relu(self.bn1(self.conv1(net), training=training))
        # net = layers.Dropout(0.3)(net)
        net = tf.nn.leaky_relu(self.bn2(self.conv2(net), training=training))
        # net = layers.Dropout(0.3)(net)
        # net = tf.nn.leaky_relu(self.bn3(self.conv3(net), training=training))
        # net = layers.Dropout(0.3)(net)
        # net = tf.nn.leaky_relu(self.bn4(self.conv4(net), training=training))
        net = self.conv6(net)
        net = tf.sigmoid(net)  # (b, 64, 64, 64, 1)

        return net


class Encoder(keras.Model):

    def __init__(self, ks=3):
        super(Encoder, self).__init__()

        self.conv1 = layers.Conv3D(128, kernel_size=ks, strides=2, padding='same')
        self.conv2 = layers.Conv3D(256, kernel_size=ks, strides=2, padding='same')
        # self.conv3 = layers.Conv3D(256, kernel_size=ks, strides=2, padding='same')
        # self.conv4 = layers.Conv3D(128, kernel_size=ks, strides=2, padding='same')
        # self.conv5 = layers.Conv3D(256, kernel_size=ks, strides=1, padding='same')
        self.conv6 = layers.Conv3D(512, kernel_size=ks, strides=2, padding='same')
        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(1024)
        self.fc2 = layers.Dense(512)
        self.noise = layers.GaussianNoise(0.1)
        self.z_mean = layers.Dense(256, name="z_mean")  # latent dim = 128
        self.z_log_var = layers.Dense(256, name="z_log_var")  # latent dim = 128

    def call(self, inputs_voxel, training=None):
        x = tf.cast(inputs_voxel, dtype=tf.float32)
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        # x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv3(x)))
        # x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv4(x)))
        # x = layers.Dropout(0.3)(tf.nn.leaky_relu(self.conv5(x)))
        x = tf.nn.leaky_relu(self.conv6(x))
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.noise(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = Sampling()([z_mean, z_log_var])

        return [z_mean, z_log_var, z]


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2, 3)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/xiaoyang/pycharm/flow_model/Dual_Encoder/datasets", help='Dataset path')
    parser.add_argument('--dual_encoder_path', type=str, default=r"/home/xiaoyang/pycharm/condition_vae/training_results/dual_encoder", help='dual encoder path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="ratio of training dataset")
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature")
    parser.add_argument('--epochs', type=int, default=2, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--query', nargs='+', default="A disordered microstructure", metavar='N', help='text query array')
    parser.add_argument('--generation_number', type=int, default=3, help="the number of generated shape for each query")
    args = parser.parse_args()
    return args


def main():

    # set up environment
    start_time = time.time()
    args = parsing()
    create_dir("training_results/vae")

    voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(dtype=float)

    images_dir = os.path.join(args.dataset_path, "figs")
    annotation_file = os.path.join(args.dataset_path, "captions.csv")
    annotations = pd.read_csv(annotation_file)["Captions"]
    image_path_to_caption = collections.defaultdict(list)

    for i in range(len(annotations)):
        caption = annotations[i]
        image_path = images_dir + "/%d" % i
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())


    print("Loading vision and text encoders...")
    vision_encoder = keras.models.load_model(os.path.join(args.dual_encoder_path, "vision_encoder"))
    text_encoder = keras.models.load_model(os.path.join(args.dual_encoder_path, "text_encoder"))
    print("Models are loaded.")

    print(f"Generating embeddings for {len(image_paths)} images...")
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(args.batch_size),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")

    # shuffle the training data
    index = np.linspace(0,1999,2000).astype(dtype=int)
    index, voxels, image_embeddings = sklearn.utils.shuffle(index, voxels, image_embeddings, random_state=0)
    voxels = np.expand_dims(voxels, -1)  # [2000, 64, 64, 64, 1]

    encoder = Encoder()
    decoder = Decoder()
    vae = VAE(encoder, decoder)

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002))
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="reconstruction_loss", factor=0.2, patience=5, min_lr=0.000001)
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss", patience=10, restore_best_weights=True)
    history = vae.fit(voxels, epochs=args.epochs, batch_size=args.batch_size, callbacks=[reduce_lr, early_stopping])  # training=False
    encoder.save("training_results/vae/encoder")
    decoder.save(("training_results/vae/decoder"))
    pd.DataFrame.from_dict(history.history).to_csv('training_results/vae/history.csv',index=False)

    # plot loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["reconstruction_loss"])
    plt.plot(history.history["kl_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train loss", "reconstruction loss", "kl_loss"], loc="upper right")
    plt.savefig("training_results/vae/loss.png")
    plt.close()

    #generate shape from query
    query = np.array(args.query)

    if not query.shape:
        query = [query]
    for i in range(len(query)):
        print(query[i])
        query_embedding = text_encoder(tf.convert_to_tensor([query[i]]))
        matches = find_matches(image_embeddings, query_embedding, normalize=True)
        matched_voxels = [voxels[idx] for idx in matches]
        matched_voxels = np.array(matched_voxels)
        [_, _, matched_voxel_embedding] = encoder(matched_voxels)
        generated_voxels = decoder(matched_voxel_embedding)
        plot_mul_3D_voxels(voxels_real=matched_voxels, voxels_fake=generated_voxels, query=query[i], save_name="training_results/vae/ground_truth_and_generated_%d" %i)

    with open('training_results/vae/accuracy.txt', 'w') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"Tensorflow version: " + tf.__version__ + "\n")
        f.write("Training time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()

