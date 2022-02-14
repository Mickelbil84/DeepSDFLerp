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
MAX_FACE_COUNT = 26000
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
