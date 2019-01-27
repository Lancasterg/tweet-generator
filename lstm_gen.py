from __future__ import absolute_import, division, print_function
import tensorflow as tf
from constants import *

tf.enable_eager_execution()

import numpy as np
import time

if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools

    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


def train_model(text):
    """
    Create, train and save the neural network model
    :return: None
    """
    # Read, then decode for py2 compat.
    # text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    print('Length of text: {} characters'.format(len(text)))
    # The unique characters in the file
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    print('{')
    for char, _ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')
    # Show how the first 13 characters from the text are mapped to integers
    print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text) // seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    # Batch size
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Training network
    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024
    BATCH_SIZE = 64

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=loss)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])
    print(history)


def loss(labels, logits):
    """
    Loss function using categorical cross entropy
    :param labels:
    :param logits:
    :return:
    """
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Build a model using Keras
    :param vocab_size: size of the vocabulary
    :param embedding_dim: dimensionality of the word embeddings
    :param rnn_units: the number of RNN nodes
    :param batch_size: size of each training batch
    :return: model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


if __name__ == '__main__':
    train_model()
