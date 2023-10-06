import json
import logging
import sys

import configargparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC

from base_optimization_operator import BaseOptimization
from data_generator import DataGeneratorMultiClass
from utils import load_model, get_training_configuration


def main():
    p = configargparse.ArgParser()
    p.add("--config-file", required=True)
    # p.add("--weights-path", required=True)
    p.add("--finetune", required=False, default=0, type=int)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    config = get_training_configuration(args.config_file)
    db_path = config.data_path
    with open(config.split_path, "r") as f:
        split = json.load(f)
    train = split["train_indexes"]
    val = split["val_indexes"]
    training_manager = DataGeneratorMultiClass(
        db_path=db_path,
        batch_size=config.batch_size,
        reference_list=train,
        data_format=config.data_format,
        seed=config.generator_seed
    )
    validation_manager = DataGeneratorMultiClass(
        db_path=db_path,
        batch_size=config.batch_size,
        reference_list=val,
        data_format=config.data_format,
        seed=config.generator_seed
    )

    classification_model = load_model("resnet.json")
    # classification_model.load_weights(args.weights_path)
    dense_layer = keras.layers.Dense(100,
                                     kernel_initializer=tf.random_normal_initializer(stddev=.01),
                                     activation="sigmoid")
    encoder_output = classification_model.get_layer(name="global_average_pooling2d").output
    output = dense_layer(encoder_output)
    classification_model = keras.Model(classification_model.input, output)
    for layer in classification_model.layers:
        if layer.name != "model_1":
            layer.trainable = args.finetune

    optimizer_param = config.optimizer_parameters[0]
    optimizer = getattr(keras.optimizers, optimizer_param["optimizer_name"])(optimizer_param["lr"])
    auc = AUC(multi_label=True)
    classification_model.compile(optimizer, "CategoricalCrossentropy",
                                 [auc, "accuracy", "top_k_categorical_accuracy"])

    optimization_operator = BaseOptimization(config, training_manager, validation_manager,
                                             config.experiment_folder)
    optimization_operator.train(classification_model)


if __name__ == "__main__":
    main()
