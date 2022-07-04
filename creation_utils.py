import inspect
import logging
import os
from typing import Type

# noinspection PyUnresolvedReferences
import tensorflow as tf
from inputtr import inputter
import tensorflow.keras.applications as architectures
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.optimizers as opts
import tensorflow.keras.layers as layer_types
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow.keras.callbacks as callbacks
from hydra.utils import get_original_cwd
from tensorflow.keras import Sequential

from arg_parser import Parser
import custom_callbacks
from config import SCHEDULER_FUNCTION_CODE_KEY, CALLBACKS_KEY, \
    FULL_TRAINING_KEY, FullConfig, SCHEDULER_VERBOSITY_KEY, SCHEDULER_FUNCTION_KEY, SCHEDULER_CONDITION_KEY, \
    KEY_WANDB, KEY_ENTITY_WANDB, KEY_PROJECT_WANDB, KEY_TENSORBOARD_CALLBACK, KEY_LOG_DIR_TBOARD, KEY_MODEL_ARCH, \
    KEY_MODEL_ARCH_NAME, KEY_MODEL_ARCH_SPECS, KEY_OPTIMIZER, KEY_LEARNING_SCHED, KEY_CUSTOM_MODEL, KEY_CUSTOM_LAYERS

BATCH_NORMALIZATION_LAYER_NAME = "BatchNormalization"
WANDB_SYNC_TBOARD = "sync_tensorboard"
CUSTOM_CALLBACK_PATH = "custom_callbacks"
DEFAULT_ARCH_NAME = "VGG19"
DEFAULT_OPT = "SGD"
DEFAULT_ARCH_SPECS = {
    "weights": "imagenet",
    "include_top": False,
    "input_shape": (224, 224, 3),
}

OPT_DICT = dict([(name, cls) for name, cls in opts.__dict__.items() if isinstance(cls, type)])
ARCH_DICT = dict([(name, cls) for name, cls in architectures.__dict__.items() if inspect.isfunction(cls)])
LAYER_CLASS_DICT = dict([(name, cls) for name, cls in layer_types.__dict__.items() if isinstance(cls, type)])
CALLBACK_DICT = dict([(name, cls) for name, cls in callbacks.__dict__.items() if isinstance(cls, type)])
CUSTOM_CB_DICT = dict([(name, cls) for name, cls in custom_callbacks.__dict__.items() if isinstance(cls, type)])
for dict_key, item in CUSTOM_CB_DICT.items():
    if "custom_callbacks" not in item.__module__:
        continue
    if dict_key in CALLBACK_DICT:
        inputter.print_warning(f"{dict_key} is overriding existing callback class!")
    CALLBACK_DICT[dict_key] = item


def default_scheduler(epoch, lr):
    return lr


def get_additional_info(cfg: FullConfig, num_classes: int) -> str:
    arch_name = DEFAULT_ARCH_NAME if KEY_MODEL_ARCH not in cfg.model else cfg.model.architecture[KEY_MODEL_ARCH_NAME]
    str_buf = f"-Model: {arch_name}\n"
    str_buf += f"-Classes: {num_classes}\n"
    str_buf += f"-Optimizer: {DEFAULT_OPT if KEY_OPTIMIZER not in cfg.hparams else cfg.hparams.optimizer}\n"
    str_buf += f"-Initial LR: {cfg.hparams.lr}\n"
    if KEY_LEARNING_SCHED in cfg.callbacks:
        str_buf += cfg.callbacks[KEY_LEARNING_SCHED]
    return str_buf


def check_wandb(cfg: FullConfig, num_classes: int):
    if KEY_WANDB in cfg:
        if KEY_ENTITY_WANDB not in cfg.wandb or KEY_PROJECT_WANDB not in cfg.wandb:
            inputter.print_warning(f"Could not initialize wandb {KEY_PROJECT_WANDB} or {KEY_ENTITY_WANDB} missing!")
            return
        run_info = f"Hydra logs: {os.getcwd()}\n"
        import wandb
        if WANDB_SYNC_TBOARD in cfg.wandb:
            for cb in cfg.callbacks:
                if KEY_TENSORBOARD_CALLBACK in cb:
                    log_dir = cb[KEY_TENSORBOARD_CALLBACK][KEY_LOG_DIR_TBOARD]
                    wandb.tensorboard.patch(
                        root_logdir=os.path.dirname(log_dir))
                    run_info += f"Tensorboard logs: {log_dir}\n"
        run_info += get_additional_info(cfg, num_classes)
        if "notes" in cfg.wandb:
            cur_notes = cfg.wandb["notes"]
            if cur_notes is None:
                cur_notes = ""
            cfg.wandb["notes"] = run_info + cur_notes
        else:
            cfg.wandb["notes"] = run_info
        run = wandb.init(**cfg.wandb)
        with open(os.path.join(get_original_cwd(), "config/full_config.yaml"), "r") as conf,\
                open(os.path.join(wandb.run.dir, "full_config.yaml"), "w") as out:
            out.write(conf.read())
        from wandb.keras import WandbCallback
        wandb_callback_class = WandbCallback
        CALLBACK_DICT[wandb_callback_class.__name__] = wandb_callback_class


def create_optimizer(cfg: FullConfig):
    if KEY_OPTIMIZER in cfg.hparams:
        opt = cfg.hparams.optimizer
        if opt in OPT_DICT:
            return OPT_DICT.get(opt)(learning_rate=cfg.hparams.lr)
    return SGD(learning_rate=cfg.hparams.lr)


# noinspection PyUnusedLocal
def test_scheduler_func(scheduler_func) -> bool:
    try:
        result = scheduler_func(1, 1, True)
        return True
    except Exception as e:
        inputter.print_error("Learning rate scheduler function failed! Aborting")
        inputter.print_error(e.__repr__())
    return False


def create_scheduler_fn(cfg: FullConfig, schedule_info: dict) -> LearningRateScheduler:
    verbosity = schedule_info[SCHEDULER_VERBOSITY_KEY] if SCHEDULER_VERBOSITY_KEY in schedule_info else 1
    if SCHEDULER_FUNCTION_KEY in schedule_info:
        func_info = schedule_info[SCHEDULER_FUNCTION_KEY]

        # noinspection PyUnusedLocal
        def scheduler(epoch, lr, should_apply: bool = False):
            init_lr = cfg.hparams.lr
            epoch_count = cfg.hparams.epochs
            if not should_apply:
                should_apply = eval(func_info[SCHEDULER_CONDITION_KEY]) if SCHEDULER_CONDITION_KEY in func_info\
                    else True
            if should_apply:
                return eval(func_info[SCHEDULER_FUNCTION_CODE_KEY])
            return lr

        if not test_scheduler_func(scheduler):
            exit(-1)
        return LearningRateScheduler(scheduler, verbose=verbosity)

    return LearningRateScheduler(default_scheduler, verbose=verbosity)


def instantiate_callbacks(cfg: FullConfig) -> list:
    parsed_callbacks = []
    if CALLBACKS_KEY not in cfg:
        return parsed_callbacks
    supplied_callbacks = cfg.callbacks
    for i in range(len(supplied_callbacks)):
        cur_cb_dict: dict = supplied_callbacks[i]
        key = list(cur_cb_dict.keys())[0]
        if key in CALLBACK_DICT:
            cb_class = CALLBACK_DICT.get(key)
            if cb_class == LearningRateScheduler:
                lr_scheduler = create_scheduler_fn(cfg, cur_cb_dict[key])
                parsed_callbacks.append(lr_scheduler)
            else:
                arg_dict = cur_cb_dict[key]
                if arg_dict is None:
                    parsed_callbacks.append(cb_class())
                else:
                    parsed_callbacks.append(cb_class(**cur_cb_dict[key]))
        else:
            logging.warning(f"Could not find Callback class for {key}")
    return parsed_callbacks


def instantiate_layer(layer_class: Type, parsed_args: dict, add_to: Model = None):
    if add_to is None:
        if len(parsed_args) == 0:
            return layer_class()
        else:
            return layer_class(**parsed_args)
    if len(parsed_args) == 0:
        return layer_class()(add_to)
    else:
        return layer_class(**parsed_args)(add_to)


def parse_custom_model(cfg: FullConfig, parser: Parser):
    layers = cfg.model.custom[KEY_CUSTOM_LAYERS]
    output_model = Sequential()
    for i in range(len(layers)):
        layer_item: dict = layers[i]
        key = list(layer_item.keys())[0]
        layer_class = LAYER_CLASS_DICT.get(list(layer_item.keys())[0])
        args: dict = layer_item[key]
        parsed_args = parser.parse_args(args) if args is not None else dict()
        output_model.add(instantiate_layer(layer_class, parsed_args))
    return output_model


def parse_model_from_config(num_classes: int, cfg: FullConfig):
    parser = Parser(cfg=cfg, num_classes=num_classes)
    if KEY_CUSTOM_MODEL in cfg.model:
        if KEY_CUSTOM_LAYERS in cfg.model.custom:
            return parse_custom_model(cfg, parser)
    train_fully = cfg.model.full_training if FULL_TRAINING_KEY in cfg.model else False

    model_class = DEFAULT_ARCH_NAME if KEY_MODEL_ARCH not in cfg.model else cfg.model.architecture[KEY_MODEL_ARCH_NAME]
    parsed_args = parser.parse_args(cfg.model.architecture[KEY_MODEL_ARCH_SPECS])\
        if KEY_MODEL_ARCH_SPECS in cfg.model.architecture else DEFAULT_ARCH_SPECS
    creation_func = ARCH_DICT[model_class]

    base_model = creation_func(**parsed_args)
    if not train_fully:
        for layer in base_model.layers:
            if BATCH_NORMALIZATION_LAYER_NAME not in layer.__class__.__name__:
                layer.trainable = False

    output_model = base_model.output
    output_layers = cfg.model.output_layers
    for i in range(len(output_layers)):
        layer_item: dict = output_layers[i]
        key = list(layer_item.keys())[0]
        layer_class = LAYER_CLASS_DICT.get(list(layer_item.keys())[0])
        args: dict = layer_item[key]
        parsed_args = parser.parse_args(args) if args is not None else dict()
        output_model = instantiate_layer(layer_class, parsed_args, output_model)
    final_model = Model(base_model.input, output_model)
    return final_model


def create_fallback_model(class_number: int) -> Model:
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        if BATCH_NORMALIZATION_LAYER_NAME not in layer.__class__.__name__:
            layer.trainable = False
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten()(head_model)
    head_model = Model(base_model.input, head_model)

    output_model = head_model.output
    output_model = Dense(256, activation="relu")(output_model)
    output_model = Dropout(0.5)(output_model)
    output_model = Dense(class_number, activation="softmax")(output_model)

    final_model = Model(base_model.input, output_model)
    return final_model
