import numpy as np
import tensorflow as tf
from autologging import logged
from tensorflow import keras

from base_optimization_operator import BaseOptimization
from data_generator import SimCLRDataGenerator
from utils import ince_loss


@logged
class SimCLROptimization(BaseOptimization):
    def __init__(self, config,
                 training_manager: SimCLRDataGenerator,
                 validation_manager: SimCLRDataGenerator,
                 experiment_folder: str = "."):
        super().__init__(config, training_manager, validation_manager, experiment_folder)
        self.model = None
        self.optimizer = None
        self._epoch = 0
        self.callback_list = []
        self.stop_training = False
        self.set_data_generator()

    def _epoch_closing(self):
        self.logs_update()
        self._print_training_logs()
        self._set_callbacks_on_epoch_end()

    def _stop_training(self) -> bool:
        stop = False
        if self.stop_training:
            stop = True
        return stop

    def _epoch_initialization(self):
        self._tr_step, self._val_step = 0, 0
        self._training_loss = []
        self._validation_loss = []
        self._set_callback_on_epoch_begin()

    def _set_callback_on_epoch_begin(self):
        for callback in self.callback_list:
            callback.on_epoch_begin(self._epoch)

    def on_train_begin(self):
        optimizer_param = self.config.optimizer_parameters
        learning_rate = optimizer_param["lr"]
        self.optimizer = getattr(keras.optimizers, optimizer_param["optimizer_name"])(
            learning_rate=learning_rate)
        self._dump_model_to_json_and_summary(self.model)
        for callback in self.callback_list:
            callback.model = self.model
            callback.model.optimizer = self.optimizer
            callback.on_train_begin()

    def fit(self, model):
        self.model = model
        self.set_callbacks()
        self.callback_list = list(self.callbacks.values())
        self.on_train_begin()
        while self._epoch < self.config.epochs:
            if self._stop_training():
                break
            self._epoch_step()
            self._epoch += 1
        self.on_train_end()
        self.close_generators()

    def _epoch_step(self):
        self._epoch_initialization()
        while self._tr_step < self.tr_steps:
            self._optimization_step()
            self._print_training_logs()
            self._tr_step += 1

            if self._tr_step >= self.tr_steps:
                self._epoch_validation()
        self._epoch_closing()

    def logs_update(self):
        self.logs["epochs"] = self._epoch
        self.logs["val_loss"] = np.mean(self._validation_loss)
        self.logs["train_loss"] = np.mean(self._training_loss)

    def _set_callbacks_on_epoch_end(self):
        for callback in self.callback_list:
            callback.on_epoch_end(epoch=self._epoch, logs=self.logs)

    def on_train_end(self):
        for callback in self.callback_list:
            callback.on_train_end()

    def _print_training_logs(self):
        self.__log.info(
            f"Epoch {self._epoch + 1}/{self.config.epochs}, batch {self._tr_step + 1}/"
            f"{self.tr_steps} "
            f"training: "
            f"loss {np.mean(self._training_loss)}")

    def _optimization_step(self):
        inputs = next(self.tr_gen)
        self._batch_optimization(inputs)

    def _epoch_validation(self):
        while self._val_step < self.val_steps:
            self._validation_step()
            self._val_step += 1
            self._print_validation_logs()

    def _print_validation_logs(self):
        self.__log.info(
            f"Epoch {self._epoch + 1}/{self.config.epochs}, validation: "
            f"loss {np.mean(self._validation_loss)}")

    def _validation_step(self):
        inputs = next(self.val_gen)
        self._batch_validation(inputs)

    def _batch_optimization(self, inputs):
        X_batch_first, X_batch_second = inputs
        with tf.GradientTape() as tape:
            z1 = self.model(X_batch_first, training=True)
            z2 = self.model(X_batch_second, training=True)
            projection_head_outputs = tf.concat((z1, z2), axis=0)
            loss = ince_loss(projection_head_outputs)
            self._training_loss.append(loss)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def _batch_validation(self, inputs):
        X_batch_first, X_batch_second = inputs
        z1 = self.model(X_batch_first, training=True)
        z2 = self.model(X_batch_second, training=True)
        projection_head_outputs = tf.concat((z1, z2), axis=0)
        loss = ince_loss(projection_head_outputs)
        self._validation_loss.append(loss)

    def np_simclr_loss(z, i, j, tau=0.5, ord = 2):
        # not working simclr loss using tf functions
        # calculates the simclr loss
        # tau is the temperature factor. In the paper, they tested values 0.1, 0.5 and 1
        # default norm is l2, no need to change that a priori

        N = int(len(z)/2)
        s = np.zeros((2*N,2*N))
        for i in range(2*N):
            for j in range(2*N):
                s[i,j] =  z[i].T@z[j]/(np.norm(z[i], ord = ord)*tf.norm(z[j], ord = ord)+1e-10)
        l = np.zeros((2*N,2*N))
        for i in range(2*N):
            for j in range(2*N):
                numerateur = np.exp(s[i,j]/tau)
                denominateur = np.sum(np.exp(s[i,:]/tau))-np.exp(s[i,i]/tau)
                l[i,j] = -np.log(numerateur/denominateur)
        loss = 1/(2*N)*np.sum(l[::2,1::2]+l[1::2,::2])
        return loss

    def tf_simclr_loss(z, i, j, tau=0.5, ord = 2):
        # not working simclr loss using tf functions
        # calculates the simclr loss
        # tau is the temperature factor. In the paper, they tested values 0.1, 0.5 and 1
        # default norm is l2, no need to change that a priori

        N = int(len(z)/2)
        s = tf.zeros([2*N,2*N])
        for i in range(2*N):
            for j in range(2*N):
            	u, v = tf.reshape(z[i], (2*N, 1)), tf.reshape(z[j], (2*N, 1))
            	s[i,j] =  tf.matmul(tf.transpose(u),v)/(tf.norm(u, ord = ord)*tf.norm(v, ord = ord)+1e-10)
        l = tf.zeros([2*N,2*N])
        for i in range(2*N):
            for j in range(2*N):
                numerateur = tf.math.exp(s[i,j]/tau)
                denominateur = tf.math.reduce_sum(tf.math.exp(s[i,:]/tau))-tf.math.exp(s[i,i]/tau)
                l[i,j] = -tf.log(numerateur/denominateur)
        loss = 1/(2*N)*tf.math.reduce_sum(l[::2,1::2]+l[1::2,::2])
        return loss


