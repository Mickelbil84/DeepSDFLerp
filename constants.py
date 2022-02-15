LATENT_DIM = 256
HIDDEN_DIM = 512
NUM_SAMPLES = 50000
ALPHA = 0.8  # percent of samples close to mesh boundary
BOUNDARY_MU = 0
BOUNDARY_SIGMA = 5e-2
NORMAL_MU = 0
NORMAL_SIGMA = 0.7
LATENT_MU = 0
LATENT_SIGMA = 0.01
MAX_FACE_COUNT = 20000000000000 #26000
NUM_THREADS = 10
NUM_MESHES = 1000  # number of meshes to take from Thingi0K


NUM_EPOCHS = 200
NUM_WORKERS = 16

BATCH_SIZE = 4096*4
LR_SDF = 1e-5
LR_EMBEDDING = 1e-3
DELTA = 0.05  # For value clamping


THINGI10K_IN_DIR = 'data/Thingi10K/raw_meshes'
THINGI10K_OUT_DIR = 'data/sampled_sdf/Thingi10K'


# Demo
MESHES = {
    "BOTTLE_CAP_TRIPOD": 0,
    "ICE_CREAM": 1,
    "VW_BUS": 2,
    "BULDOZER": 15,
    "HIVE": 49,
    "HURRICANE_LAMP": 73,
    "LEGO": 99,
    "BIG_BOY_LOCOMOTIVE": 137,
    "PONYTROPE": 320,
    "LYMAN_FILLAMENT_EXTRUDER": 402,
}


CHECKPOINT_DEEPSDF = 'resources/trained_models/embedding_on_1000/checkpoint_e199.pth'
CHECKPOINT_EMBEDDING = 'resources/trained_models/embedding_on_1000/checkpoint_e199_emb.pth'


# Run the model trained on 100 meshes instead of 1000
SMALLER_MODEL = True
if SMALLER_MODEL:
    CHECKPOINT_DEEPSDF = 'resources/trained_models/embedding_on_100/checkpoint_e98.pth'
    CHECKPOINT_EMBEDDING = 'resources/trained_models/embedding_on_100/checkpoint_e98_emb.pth'
    MESHES = {key: value for key, value in MESHES.items() if value < 100}
    NUM_MESHES = 100
