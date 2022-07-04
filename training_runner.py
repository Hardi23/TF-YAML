import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model
# from tensorflow.python import keras
from tensorflow import keras
import creation_utils
from config import FullConfig, MODEL_KEY, KEY_CUSTOM_LAYERS, KEY_MODEL_PREVIEW, KEY_OPTIONS

DATE_STRING = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def fit_like_generator(
                    model: tf.keras.Model,
                    train_gen: ImageDataGenerator,
                    validation_gen: ImageDataGenerator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0):
    model.fit(x=train_gen, validation_data=validation_gen, verbose=verbose,
              callbacks=callbacks, validation_split=validation_gen, epochs=epochs,
              steps_per_epoch=steps_per_epoch, class_weight=class_weight, max_queue_size=max_queue_size,
              workers=workers, use_multiprocessing=use_multiprocessing, shuffle=shuffle, initial_epoch=initial_epoch,
              validation_steps=validation_steps, validation_freq=validation_freq)


def load_model(cfg: FullConfig) -> keras.Model:
    model_path: str = cfg.model.file
    print(f"[INFO] - Loading model from file: {model_path}")
    if model_path.endswith(".json"):
        return keras.models.model_from_json(model_path)
    elif model_path.endswith(".h5") or model_path.endswith(".keras"):
        return keras.models.load_model(cfg.model.file)


def configure_gpus():
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)


def write_mapping(cfg: FullConfig):
    mapping_file = cfg.paths.mapping_file
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    class_strs = os.listdir(cfg.paths.training)
    class_strs.sort(key=str)
    with open(mapping_file, "w", encoding="utf-8") as f:
        counter: int = 0
        for clazz in class_strs:
            f.write(f"{counter} {clazz}\n")
            counter += 1
        f.flush()


def log_model(model):
    logging.info("====== Created model ======")
    buf = []
    model.summary(print_fn=buf.append)
    logging.info("\n".join(buf))


class TrainingRunner:
    __config: FullConfig
    __cwd: str

    def __init__(self, cfg: FullConfig):
        self.__config = cfg

    def run(self) -> None:
        optimizer = creation_utils.create_optimizer(self.__config)

        img_data_gen = ImageDataGenerator(validation_split=self.__config.hparams.val_split)
        img_data_gen.mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        train_data_gen = img_data_gen.flow_from_directory(
            self.__config.paths.training,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            seed=42,
            batch_size=self.__config.hparams.batch_size,
            subset="training"
        )
        val_data_gen = img_data_gen.flow_from_directory(
            self.__config.paths.training,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            seed=42,
            batch_size=self.__config.hparams.batch_size,
            subset="validation"
        )
        write_mapping(self.__config)
        configure_gpus()
        model: Optional[Model] = None
        if MODEL_KEY in self.__config:
            if self.__config.model and "file" in self.__config.model:
                model = load_model(self.__config)
            elif "output_layers" in self.__config.model or KEY_CUSTOM_LAYERS in self.__config.model:
                model = creation_utils.parse_model_from_config(train_data_gen.num_classes, self.__config)

        if not model:
            if KEY_OPTIONS in self.__config and self.__config.opt.fallback:
                print("Error creating model defaulting to fallback model")
                model = creation_utils.create_fallback_model(train_data_gen.num_classes)
            else:
                print("Error creating model exiting")
                exit(1)

        log_model(model)
        if KEY_OPTIONS in self.__config:
            if KEY_MODEL_PREVIEW in self.__config.opt and self.__config.opt.preview:
                exit(0)
        creation_utils.check_wandb(self.__config, train_data_gen.num_classes)
        callbacks_list = creation_utils.instantiate_callbacks(self.__config)

        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=['accuracy', 'categorical_accuracy'])
        model.fit_generator(
            train_data_gen,
            verbose=1,
            steps_per_epoch=train_data_gen.samples // self.__config.hparams.batch_size,
            validation_data=val_data_gen,
            validation_steps=val_data_gen.samples // self.__config.hparams.batch_size,
            epochs=self.__config.hparams.epochs,
            callbacks=callbacks_list
        )
