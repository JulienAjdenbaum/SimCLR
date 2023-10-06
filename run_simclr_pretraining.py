import json
import logging
import sys

import configargparse
import numpy as np

from data_generator import SimCLRDataGenerator
from perturbator.composition import ComposePerturbator
from perturbator.crop import Crop
from perturbator.flip import Flip
from perturbator.gaussian_blur import GaussianBlurPerturbator
from perturbator.rotation import Rotation
from simclr_optimization import SimCLROptimization
from utils import get_training_configuration, load_model


def main():
    p = configargparse.ArgParser()
    p.add("--config-file", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    config = get_training_configuration(args.config_file)
    root_folder = "."
    db_path = config.data_path
    with open(config.split_path, "r") as f:
        split = json.load(f)
    train = split["train_indexes"]
    val = split["val_indexes"]
    np.random.shuffle(train)
    np.random.shuffle(val)

    ### you can change the perturbation used and code new perturbations if needed
    # such as color jittering
    perturbator = ComposePerturbator([GaussianBlurPerturbator(prob=0.5, sigma_range=[0.1, 2.0]),
                                      Crop(prob=0.5, crop_size_ratio=[0.5, 0.5],
                                           cropping_type="random", resize=True, keep_dim=True),
                                      Flip(prob=0.5),
                                      Rotation(prob=0.5)])
    training_manager = SimCLRDataGenerator(db_path=db_path,
                                           perturbator=perturbator,
                                           batch_size=config.batch_size,
                                           reference_list=train,
                                           data_format=config.data_format,
                                           seed=config.generator_seed,
                                           perturbate_second=config.perturbate_second
                                           )
    validation_manager = SimCLRDataGenerator(db_path=db_path,
                                             perturbator=perturbator,
                                             batch_size=config.batch_size,
                                             reference_list=val,
                                             data_format=config.data_format,
                                             seed=config.generator_seed,
                                             perturbate_second=config.perturbate_second
                                             )

    simclr_encoder = load_model(f"{root_folder}/resnet.json")

    optimization_operator = SimCLROptimization(config, training_manager, validation_manager,
                                               config.experiment_folder)
    optimization_operator.fit(simclr_encoder)


if __name__ == "__main__":
    main()
