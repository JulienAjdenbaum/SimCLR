from typing import List

import h5py
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class DataGenerator:

    def __init__(self,
                 db_path,
                 batch_size,
                 reference_list: List[List[str]],
                 data_format="channels_first",
                 seed=0):
        np.random.seed(seed)
        self._db = h5py.File(db_path, "r")
        self._reference_list = reference_list
        self.num_index = len(reference_list)
        self._index_list = np.arange(0, self.num_index)
        self._batch_size = batch_size
        self.passes = np.Inf
        self.data_format = data_format

    def _generator(self):
        epochs = 0
        while epochs < self.passes:
            self._shuffle_index_list()
            for index in np.arange(0, self.num_index, self._batch_size):
                x_list = []
                y_list = []
                for i in self._index_list[index:index + self._batch_size]:
                    reference = self._reference_list[i]
                    data = self._get_data(reference)
                    new_annotation = self._get_annotation(reference)
                    x_list.append(data)
                    y_list.append(new_annotation)

                x_batch = self.reformat_input_for_keras_model(x_list,
                                                              self.data_format)
                y_batch = np.array(y_list)
                yield x_batch, y_batch
            epochs += 1

    @staticmethod
    def reformat_input_for_keras_model(x, data_format="channels_first"):
        x = np.array(x)
        if len(x.shape) < 4:
            if data_format == "channels_first":
                x = x[:, np.newaxis, ...]
            else:
                x = x[:, ..., np.newaxis]
        return x

    def _shuffle_index_list(self):
        np.random.shuffle(self._index_list)

    def close(self):
        self._db.close()

    def _get_data(self, reference):
        return self._db[f"data/{reference[0]}/{reference[1]}/data"][()]

    def _get_annotation(self, reference):
        return self._db[f"data/{reference[0]}/{reference[1]}/label/classification"][()]

    def get_generator(self):
        return self._generator()


class SimCLRDataGenerator(DataGenerator):
    """
    """

    def __init__(self, db_path,
                 batch_size,
                 reference_list: List[List[str]],
                 perturbator,
                 data_format="channels_first",
                 seed=0,
                 perturbate_second=False):
        super(SimCLRDataGenerator, self).__init__(db_path=db_path, batch_size=batch_size,
                                                  reference_list=reference_list,
                                                  data_format=data_format, seed=seed)
        self.perturbate_second = perturbate_second
        self.perturbator = perturbator

    def _generator(self):
        epochs = 0
        while epochs < self.passes:
            self._shuffle_index_list()
            for index in np.arange(0, self.num_index, self._batch_size):
                x1_batch, x2_batch = self.get_x1_x2_batch(index)
                yield x1_batch, x2_batch
            epochs += 1

    def _get_data_and_perturbate(self, reference):
        data = self._get_data(reference)
        return self.perturbator(data)

    def get_x1_x2_batch(self, index):
        x1_list = []
        x2_list = []
        for i in self._index_list[index:index + self._batch_size]:
            reference = self._reference_list[i]
            first_perturbated_data = self._get_data_and_perturbate(reference)
            if self.perturbate_second:
                second_perturbated_data = self._get_data_and_perturbate(reference)
            else:
                second_perturbated_data = self._get_data(reference)
            x1_list.append(first_perturbated_data)
            x2_list.append(second_perturbated_data)
        x1_batch = self.reformat_input_for_keras_model(x1_list)
        x2_batch = self.reformat_input_for_keras_model(x2_list)
        return x1_batch, x2_batch


class DataGeneratorMultiClass(DataGenerator):
    def _get_annotation(self, reference):
        label = self._db[f"data/{reference[0]}/{reference[1]}/label/classification"][()]
        mlb = MultiLabelBinarizer()
        mlb.fit([[i] for i in range(100)])
        return mlb.transform([[label]])[0].astype(np.float32)
