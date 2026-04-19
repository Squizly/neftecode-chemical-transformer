import torch

PATH_RAW_TRAIN = 'data/raw/daimler_mixtures_train.csv'
PATH_RAW_TEST = 'data/raw/daimler_mixtures_test.csv'
PATH_RAW_PROPS = 'data/raw/daimler_component_properties.csv'

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

TARGET_COLS_INTERNAL = ["target_visc", "target_oxid"]
TARGET_COLS_SUBMISSION = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
]
DOSE_COL, TEMP_COL, TIME_COL, BIO_COL, CAT_COL = "mass_norm", "temp", "time", "biofuel", "catalyst"

# --- ПАРАМЕТРЫ АРХИТЕКТУРЫ ---
D_MODEL = 32
N_HEADS = 4 
N_LAYERS = 2 
D_FF = 64 
DROPOUT = 0.25 
N_SEEDS = 3

# --- ОБУЧЕНИЕ ---
EPOCHS = 1000
LR = 2e-4
WEIGHT_DECAY = 1e-2 
PATIENCE = 150 
BATCH_SIZE = 16
SEED = 42 
N_ENSEMBLE_SEEDS = 1

# --- АУГМЕНТАЦИЯ ---
AUG_NOISE_STD = 0.03
AUG_DROP_PROB = 0.05
AUG_DROP_THRESHOLD = 0.1
AUG_MULTIPLIER = 5
AUG_DOSE_JITTER = 0.015