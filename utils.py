import json

import ml_collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


def load_model(model_json_file) -> tf.keras.Model:
    json_file = open(model_json_file, 'r')
    model_json = json_file.read()
    json_file.close()
    return model_from_json(model_json)


def ince_loss(input_data, temperature=1.0, large_num=1e9) -> np.float32:
    hidden = input_data
    hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * large_num
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature

    logits_a = tf.concat([logits_ab, logits_aa], 1)
    logits_b = tf.concat([logits_ba, logits_bb], 1)

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, logits_a)
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, logits_b)
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss


def get_training_configuration(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return ml_collections.ConfigDict(config)
