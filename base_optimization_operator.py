import os

from tensorflow import keras

from data_generator import DataGenerator


class BaseOptimization:
    def __init__(self,
                 config,
                 training_manager: DataGenerator,
                 validation_manager: DataGenerator,
                 experiment_folder: str = "."):
        self.config = config
        self.training_manager = training_manager
        self.validation_manager = validation_manager
        self.tr_steps = 0
        self.val_steps = 0
        self.tr_gen = None
        self.val_gen = None
        self.experiment_folder = experiment_folder
        self.callbacks = {}

    def set_data_generator(self):
        self.tr_steps = self.training_manager.num_index // self.config.batch_size
        self.val_steps = self.validation_manager.num_index // self.config.batch_size
        self.tr_gen = self.training_manager.get_generator()
        self.val_gen = self.validation_manager.get_generator()

    def set_callbacks(self):
        call_backs_dict = self.config["callback_list"]
        for fct, params in call_backs_dict.items():
            if fct not in dir(keras.callbacks):
                Warning("Did not find {} in keras.callbacks.".
                        format(fct))
                continue
            if fct == "ModelCheckpoint":
                params["filepath"] = os.path.join(
                    f"{self.experiment_folder}/weights", 'weights_{epoch:02d}.hdf5')

            if fct == "CSVLogger":
                params["filename"] = os.path.join(self.experiment_folder, "log.csv")
            self.callbacks[fct] = getattr(keras.callbacks, fct)(**params)

    def _dump_model_to_json_and_summary(self,
                                        model: keras.Model,
                                        name: str = "model"):
        model_json = model.to_json()
        with open(os.path.join(self.experiment_folder, name + ".json"), "w") as json_file:
            json_file.write(model_json)
        write_path = os.path.join(
            self.experiment_folder, name + "_summary.txt")
        with open(write_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    def close_generators(self):
        self.training_manager.close()
        self.validation_manager.close()

    def train(self, model):
        self.set_data_generator()
        self.set_callbacks()
        self._dump_model_to_json_and_summary(model)
        model.fit(
            self.tr_gen,
            steps_per_epoch=self.tr_steps,
            validation_data=self.val_gen,
            validation_steps=self.val_steps,
            epochs=self.config.epochs,
            callbacks=list(self.callbacks.values())
        )
        self.close_generators()
