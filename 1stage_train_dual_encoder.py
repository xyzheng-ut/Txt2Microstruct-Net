import os
import sys
sys.path.append(r"/home/xiaoyang/anaconda3/lib/python3.11/site-packages")
proxy = 'http://proxyout.nims.go.jp:8888'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import collections
import json
import numpy as np
import datetime
import flatbuffers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import pandas as pd
import random
import argparse
import time


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not osp.exists(path):
        os.makedirs(path)


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(image_path, caption):
    feature = {
        "caption": bytes_feature(caption.encode()),
        "raw_image_0": bytes_feature(tf.io.read_file(image_path+"_0.jpg").numpy()),
        "raw_image_1": bytes_feature(tf.io.read_file(image_path+"_1.jpg").numpy()),
        "raw_image_2": bytes_feature(tf.io.read_file(image_path+"_2.jpg").numpy())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, image_paths, image_path_to_caption):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:1]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example(
                image_path_list[example_idx], caption_list[example_idx]
            )
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(image_paths, num_files, files_prefix, image_path_to_caption, images_per_file=2000):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx], image_path_to_caption)
    return example_counter


def read_example(example):
    feature_description = {
        "caption": tf.io.FixedLenFeature([], tf.string),
        "raw_image_0": tf.io.FixedLenFeature([], tf.string),
        "raw_image_1": tf.io.FixedLenFeature([], tf.string),
        "raw_image_2": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)
    raw_image_0 = features.pop("raw_image_0")
    raw_image_1 = features.pop("raw_image_1")
    raw_image_2 = features.pop("raw_image_2")
    features["image"] = tf.image.resize(
        tf.concat([tf.image.decode_jpeg(raw_image_0, channels=1),
                  tf.image.decode_jpeg(raw_image_1, channels=1),
                  tf.image.decode_jpeg(raw_image_2, channels=1)], axis=-1
                  ), size=(299, 299)
    )
    return features


def get_dataset(file_pattern, batch_size):

    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .batch(batch_size)
    )


def project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


def create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained Xception model to be used as the base encoder.
    xception = keras.applications.Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in xception.layers:
        layer.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input(shape=(299, 299, 3), name="image_input")
    # Preprocess the input image.
    xception_input = tf.keras.applications.xception.preprocess_input(inputs)
    # Generate the embeddings for the images using the xception model.
    embeddings = xception(xception_input)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(inputs, outputs, name="vision_encoder")


def create_text_encoder(
        num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the BERT preprocessing module.
    # BERT preprocessing module can be downloaded from the internet or local file
    # preprocess = hub.KerasLayer(
    #     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    #     name="text_preprocessing",
    #     proxy="http://proxyout.nims.go.jp:8888"
    # )
    #
    preprocess = keras.models.load_model(r"pretrained_models/text_preprocessing")
    # Load the pre-trained BERT model to be used as the base encoder.
    # bert = hub.KerasLayer(
    #     "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
    #     "bert",
    #     proxy="http://proxyout.nims.go.jp:8888"
    # )
    bert = hub.KerasLayer(r"pretrained_models/small_bert",
                          name="bert")
    # bert = keras.models.load_model("pretrained_models/small_bert")
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")


class DualEncoder(keras.Model):
    def __init__(self, text_encoder, vision_encoder, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        with tf.device("/gpu:0"):
            # Get the embeddings for the captions.
            caption_embeddings = self.text_encoder(features["caption"], training=training)
        with tf.device("/gpu:1"):
            # Get the embeddings for the images.
            image_embeddings = self.vision_encoder(features["image"], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = (
                tf.matmul(caption_embeddings, image_embeddings, transpose_b=True)
                / self.temperature
        )
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = tf.matmul(
            image_embeddings, image_embeddings, transpose_b=True
        )
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = tf.matmul(
            caption_embeddings, caption_embeddings, transpose_b=True
        )
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the captions using crossentropy
        captions_loss = keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the images using crossentropy
        images_loss = keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def read_image(image_path):
    image_path_0 = image_path+"_0.jpg"
    image_path_1 = image_path+"_1.jpg"
    image_path_2 = image_path+"_2.jpg"
    image_array = tf.concat((tf.image.decode_jpeg(tf.io.read_file(image_path_0), channels=1),
                              tf.image.decode_jpeg(tf.io.read_file(image_path_1), channels=1),
                              tf.image.decode_jpeg(tf.io.read_file(image_path_2), channels=1)),
                             axis=-1)
    return tf.image.resize(image_array, (299, 299))


def find_matches(image_embeddings, queries, text_encoder, image_paths, k=9, normalize=True):
    # Get the embedding for the query.
    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    # Normalize the query and the image embeddings.
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    # Compute the dot product between the query and the image embeddings.
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    # Retrieve top k indices.
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    # Return matching image paths.
    return [[image_paths[idx] for idx in indices] for indices in results]


def compute_top_k_accuracy(image_paths, batch_size, image_path_to_caption, image_embeddings, text_encoder, all_image_paths, k=100):
    hits = 0
    num_batches = int(np.ceil(len(image_paths) / batch_size))
    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        current_image_paths = image_paths[start_idx:end_idx]
        queries = [
            image_path_to_caption[image_path][0] for image_path in current_image_paths
        ]
        result = find_matches(image_embeddings, queries, text_encoder, all_image_paths, k)
        hits += sum(
            [
                image_path in matches
                for (image_path, matches) in list(zip(current_image_paths, result))
            ]
        )

    return hits / len(image_paths)


def parsing(mode="args"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/xiaoyang/pycharm/flow_model/Dual_Encoder/datasets", help='Dataset path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="ratio of training dataset")
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature")
    parser.add_argument('--epochs', type=int, default=200, help="training epoch")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--query', type=str, default="Micrograph shows structure comprising polygonal ferrite grains. Irregular Voronoi cells define the spatial distribution.", help="text query")
    args = parser.parse_args()
    return args


def main():

    # set up environment
    start_time = time.time()
    args = parsing()
    create_dir("training_results/dual_encoder")
    create_dir(osp.join(args.dataset_path, "tfrecords"))

    random.seed(22)
    # Suppressing tf.hub warnings
    tf.get_logger().setLevel("ERROR")
    print(tf.__version__)
    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("tensorflow is running on GPU:", tf.config.list_physical_devices('GPU'))
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # load dataset
    images_dir = osp.join(args.dataset_path, "figs")
    tfrecords_dir = osp.join(args.dataset_path, "tfrecords")
    annotation_file = osp.join(args.dataset_path, "captions.csv")


    annotations = pd.read_csv(annotation_file)["Captions"]

    image_path_to_caption = collections.defaultdict(list)

    for i in range(len(annotations)):
        caption = annotations[i]
        image_path = images_dir + "/%d" % i
        image_path_to_caption[image_path].append(caption)

    items = list(image_path_to_caption.items())
    random.shuffle(items)
    image_path_to_caption = collections.OrderedDict(items)

    image_paths = list(image_path_to_caption.keys())
    print(f"Number of images: {len(image_paths)}")
    images_per_file = len(image_paths)
    train_size = np.round(args.split_ratio*images_per_file).astype(dtype=int)
    valid_size = len(image_paths) - train_size

    train_image_paths = image_paths[:train_size]
    num_train_files = int(np.ceil(train_size / images_per_file))
    train_files_prefix = os.path.join(tfrecords_dir, "train")

    valid_image_paths = image_paths[-valid_size:]
    num_valid_files = int(np.ceil(valid_size / images_per_file))
    valid_files_prefix = os.path.join(tfrecords_dir, "valid")

    train_example_count = write_data(train_image_paths, num_train_files, train_files_prefix, image_path_to_caption, images_per_file)
    print(f"{train_example_count} training examples were written to tfrecord files.")
    valid_example_count = write_data(valid_image_paths, num_valid_files, valid_files_prefix, image_path_to_caption, images_per_file)
    print(f"{valid_example_count} evaluation examples were written to tfrecord files.")

    tf.io.gfile.makedirs(tfrecords_dir)

    num_epochs = args.epochs
    batch_size = args.batch_size
    temperature = args.temperature
    vision_encoder = create_vision_encoder(
        num_projection_layers=1, projection_dims=256, dropout_rate=0.1, trainable=False
    )
    text_encoder = create_text_encoder(
        num_projection_layers=1, projection_dims=256, dropout_rate=0.1, trainable=False
    )
    dual_encoder = DualEncoder(text_encoder, vision_encoder, temperature=temperature)
    dual_encoder.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001)
    )

    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {int(np.ceil(train_example_count / batch_size))}")
    train_dataset = get_dataset(os.path.join(tfrecords_dir, "train-*.tfrecord"), batch_size)
    valid_dataset = get_dataset(os.path.join(tfrecords_dir, "valid-*.tfrecord"), batch_size)
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = dual_encoder.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=[reduce_lr, early_stopping],
    )
    print("Training completed. Saving vision and text encoders...")
    vision_encoder.save("training_results/dual_encoder/vision_encoder")
    text_encoder.save("training_results/dual_encoder/text_encoder")
    print("Models are saved.")

    pd.DataFrame.from_dict(history.history).to_csv('training_results/dual_encoder/history.csv', index=False)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "valid"], loc="upper right")
    plt.savefig(r"training_results/dual_encoder/loss.png")
    plt.close()

    print(f"Generating embeddings for {len(image_paths)} images...")
    image_embeddings = vision_encoder.predict(
        tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
        verbose=1,
    )
    print(f"Image embeddings shape: {image_embeddings.shape}.")

    query = args.query
    matches = find_matches(image_embeddings, [query], text_encoder, image_paths, normalize=True)[0]

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(query, fontsize=24)
    for i in range(4):
        for j in range(3):
            ax = plt.subplot(4, 3, i*3 + j + 1)
            img_path = matches[i]+"_%d.jpg" %j
            plt.imshow(mpimg.imread(img_path))
            plt.axis("off")

    plt.savefig("training_results/dual_encoder/text_retrieval.png")
    plt.close()

    print("Scoring training data...")

    train_accuracy = compute_top_k_accuracy(train_image_paths, batch_size, image_path_to_caption, image_embeddings, text_encoder, image_paths)
    print(f"Train accuracy: {round(train_accuracy * 100, 3)}%")

    print("Scoring evaluation data...")
    eval_accuracy = compute_top_k_accuracy(image_paths[train_size:], batch_size, image_path_to_caption, image_embeddings, text_encoder, image_paths)
    print(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%")

    with open('training_results/dual_encoder/accuracy.txt', 'w') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(f"Tensorflow version: " + tf.__version__ + "\n")
        f.write(f"Numbers of training dataset: %d\n" % train_size)
        f.write(f"Numbers of validating dataset: %d\n" % valid_size)
        f.write(f"Train accuracy: {round(train_accuracy * 100, 3)}%\n")
        f.write(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%\n")
        f.write("Training time: %s seconds" % (time.time() - start_time))
        # f.write("training time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":
    main()