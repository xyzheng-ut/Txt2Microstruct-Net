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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    plt.close()


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
                    keras.losses.mae(x_mean, z_mean), axis=0
                ) +
                tf.reduce_sum(
                    keras.losses.mae(x_log_var, z_log_var), axis=0
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


def preprocess(voxels):
    voxels = tf.cast(voxels, dtype=tf.float32)
    voxels = tf.expand_dims(voxels, -1)  # [2000, 64, 64, 64, 1]
    return voxels


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/xiaoyang/pycharm/flow_model/Dual_Encoder/datasets", help='Dataset path')
    parser.add_argument('--dual_encoder_path', type=str, default=r"/home/xiaoyang/pycharm/condition_vae/training_results/dual_encoder", help='dual encoder path')
    parser.add_argument('--vae_path', type=str, default=r"/home/xiaoyang/pycharm/condition_vae/training_results/vae", help='vae path')
    parser.add_argument('--query', nargs='+', default="Randomly shaped polyhedral voids within the closed-cell foam network.", metavar='N', help='text query array')
    parser.add_argument('--generation_number', type=int, default=3, help="the number of generated shape for each query")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=500, help="training epoch")
    parser.add_argument('--latent_dim', type=int, default=256, help="latent dim")
    args = parser.parse_args()
    return args


def main():
    args = parsing()
    create_dir("training_results/mlp")
    batch_size = args.batch_size
    latent_dim = args.latent_dim
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

    print("Loading VAE...")
    encoder = keras.models.load_model(os.path.join(args.vae_path, "encoder"))
    decoder = keras.models.load_model(os.path.join(args.vae_path, "decoder"))
    print("VAE model are loaded.")

    print(f"Generating embeddings for {len(image_paths)} images...")
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(args.batch_size),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")

    # shuffle the training data
    # index = np.linspace(0,1999,2000).astype(dtype=int)
    # index, voxels, image_embeddings = sklearn.utils.shuffle(index, voxels, image_embeddings, random_state=0)
    # voxels = np.expand_dims(voxels, -1)  # [2000, 64, 64, 64, 1]
    # voxels = tf.convert_to_tensor(voxels)
    dataset = tf.data.Dataset.from_tensor_slices(voxels).map(preprocess).batch(args.batch_size)
    for step, voxel_batch in enumerate(dataset):
        [x_mean_batch, x_log_var_batch, _] = encoder(voxel_batch)

        if step == 0:
            x_mean = x_mean_batch
            x_log_var = x_log_var_batch
        else:
            x_mean = tf.concat((x_mean, x_mean_batch), axis=0)
            x_log_var = tf.concat((x_log_var, x_log_var_batch), axis=0)


    print(tf.reduce_max(x_mean).numpy())
    print(tf.reduce_min(x_mean).numpy())
    print(tf.reduce_max(x_log_var).numpy())
    print(tf.reduce_min(x_log_var).numpy())
    x_mean = x_mean/2.5
    x_log_var = x_log_var/-10
    # x_mean = (x_mean - x_mean_min)/(x_mean_max - x_mean_min)
    # x_log_var = (x_log_var - x_log_var_min) / (x_log_var_max - x_log_var_min)
    x_mean_x_log_var = tf.concat((x_mean, x_log_var), axis=-1)
    # norm = layers.Normalization()
    # norm.adapt(image_embeddings)
    # norm2 = layers.Normalization()
    # norm2.adapt(x_mean_x_log_var)
    # #
    # image_embeddings = norm(image_embeddings)
    # x_mean_x_log_var = norm2(x_mean_x_log_var)

    #
    # print(normalized_data)
    # print(normalized_data.shape)
    # prepare text embedding
    caption_embeddings = text_encoder(tf.convert_to_tensor(annotations))
    print("caption_embeddings", caption_embeddings.shape)
    for i in range(len(caption_embeddings)):
        matches = find_matches(image_embeddings, [caption_embeddings[i]], normalize=True)
        matched_x_mean_x_log_var = [x_mean_x_log_var[idx] for idx in matches]
        matched_x_mean_x_log_var = np.expand_dims(np.array(matched_x_mean_x_log_var), axis=0)
        if i ==0:
            txt_x_mean_x_log_var = matched_x_mean_x_log_var
        else:
            txt_x_mean_x_log_var = tf.concat((txt_x_mean_x_log_var, matched_x_mean_x_log_var), axis=0)

    print("txt_x_mean_x_log_var", txt_x_mean_x_log_var.shape)

    # dataset = tf.data.Dataset.from_tensor_slices((image_embeddings, txt_x_mean_x_log_var))
    dataset = tf.data.Dataset.from_tensor_slices((caption_embeddings, txt_x_mean_x_log_var))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Creating a custom layer with keras API.

    mlp = MLP()

    mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003)
    )

    history = mlp.fit(dataset, epochs=args.epochs)

    mlp.save("training_results/mlp/mlp")

    plt.figure(figsize=(15, 10))
    plt.plot(history.history["loss"])
    # plt.plot(history.history["val_loss"])
    plt.title("mlp loss")
    plt.legend(["loss"], loc="upper right")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("training_results/mlp/mlp_loss.png")
    plt.close()
    # plt.show()


    # From data to latent space.

    #generate shape from query
    query = np.array(args.query)

    if not query.shape:
        query = [query]
    for i in range(len(query)):
        query_embedding = text_encoder(tf.convert_to_tensor([query[i]]))
        # norm3 = layers.Normalization()
        # norm3.adapt(query_embedding)
        # query_embedding = norm(query_embedding)
        # query_embedding = tf.tile(query_embedding, [3,1])
        # noise = tf.random.normal(shape=[len(query_embedding), 256])
        # noise_query_embedding = tf.concat([noise, query_embedding], axis=1)
        [y_mean, y_log_var, _] = mlp(query_embedding)
        # y_mean = (x_mean_max - x_mean_min)*y_mean + x_mean_min
        # y_log_var = (x_log_var_max - x_log_var_min) * y_log_var + x_log_var_min
        y_mean = tf.reshape(y_mean, shape=[-1, 256])
        y_log_var = tf.reshape(y_log_var, shape=[-1, 256])
        y_mean = y_mean * 2.5
        y_log_var = y_log_var*-10
        y = Sampling()([y_mean, y_log_var])
        generated_voxels = decoder(y)
        plot_mul_3D_voxels(generated_voxels[:3], query=query[i], save_name="training_results/mlp/ground_truth_and_generated_%d" %i)

        # [y_mean, y_log_var,_] = mlp(image_embeddings[:3])
        # y_mean = (x_mean_max - x_mean_min)*y_mean + x_mean_min
        # y_log_var = (x_log_var_max - x_log_var_min) * y_log_var + x_log_var_min
        # y = Sampling()([y_mean, y_log_var])
        # generated_voxels2 = decoder(y)
        # plot_mul_3D_voxels(generated_voxels2, query="from image embeddings", save_name="training_results/mlp/from image embedding_%d" %i)


if __name__ == '__main__':
    main()