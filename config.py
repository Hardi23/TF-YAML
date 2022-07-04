from dataclasses import dataclass, field
CONFIG_PATH = "config"
CONFIG_NAME = "full_config.yaml"
SCHEDULER_VERBOSITY_KEY = "verbose"
SCHEDULER_FUNCTION_KEY = "function"
SCHEDULER_CONDITION_KEY = "condition"
SCHEDULER_FUNCTION_CODE_KEY = "rate_function"
CALLBACKS_KEY = "callbacks"
REPLACER_CLASS_COUNT = "??classes"
FULL_TRAINING_KEY = "full_training"
KEY_OPTIMIZER = "optimizer"
KEY_WANDB = "wandb"
KEY_PROJECT_WANDB = "project"
KEY_ENTITY_WANDB = "entity"
KEY_TENSORBOARD_CALLBACK = "TensorBoard"
KEY_LOG_DIR_TBOARD = "log_dir"
KEY_MODEL_ARCH = "architecture"
KEY_MODEL_ARCH_NAME = "name"
KEY_MODEL_ARCH_SPECS = "specs"
KEY_LEARNING_SCHED = "LearningRateScheduler"
KEY_CUSTOM_MODEL = "custom"
KEY_CUSTOM_LAYERS = "layers"
KEY_OPTIONS = "opt"
KEY_MODEL_PREVIEW = "preview"

MODEL_KEY = "model"
OPT_ADAM = "adam"
OPT_SGD = "sgd"


@dataclass
class Opt:
    preview: bool
    fallback: bool


@dataclass
class Model:
    architecture: dict
    full_training: bool
    file: str
    custom: dict[list[dict]]
    output_layers: list = field(default_factory=list)


@dataclass
class Schedule:
    dec: str
    amount: float
    start_epoch: int


@dataclass
class HParams:
    lr: float
    optimizer: str
    epochs: int
    batch_size: int
    val_split: float


@dataclass
class Paths:
    training: str
    test: str
    mapping_file: str


@dataclass
class FullConfig:
    hparams: HParams
    paths: Paths
    model: Model
    opt: Opt
    wandb: dict = field(default_factory=dict)
    callbacks: list = field(default_factory=list)
