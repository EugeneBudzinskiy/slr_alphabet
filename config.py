# ===============
#  GLOBAL PARAMS
# ===============

MODEL_PATH = "model.h5"
IMAGE_SIZE = 200  # W: 200px | H: 200px
IMAGE_CHANNEL = 3  # 3 color channels

BATCH_SIZE = 30
NUMBER_OF_CLASSES = 27  # Number of classes to predict
N_FRAMES_PER_SECOND = 6  # Process only N frames per second from video feed



# ===============
#  TRAIN DATASET
# ===============

RAW_TRAIN_DATASET_PATH = "dataset_raw/Train_Alphabet"  # Path to folder with train images (in class folders)
TRAIN_N_IMAGES_PER_FOLDER = 200  # Get random N images in each class folder
TRAIN_IMAGE_TOTAL_NUMBER = TRAIN_N_IMAGES_PER_FOLDER * NUMBER_OF_CLASSES

TRAIN_DATA_MEMMAP_PATH = "dataset/train/data.dat"
TRAIN_LABEL_NPY_PATH = "dataset/train/label.npy"
TRAIN_LABEL_DECODE_JSON_PATH = "dataset/train/label_decode.json"



# ===============
#  TEST DATASET
# ===============

RAW_TEST_DATASET_PATH = "dataset_raw/Test_Alphabet"  # Path to folder with test images (in class folders)
TEST_N_IMAGES_PER_FOLDER = 50  # Get random N images in each class folder
TEST_IMAGE_TOTAL_NUMBER = TEST_N_IMAGES_PER_FOLDER * NUMBER_OF_CLASSES

TEST_DATA_MEMMAP_PATH = "dataset/test/data.dat"
TEST_LABEL_NPY_PATH = "dataset/test/label.npy"
TEST_LABEL_DECODE_JSON_PATH = "dataset/test/label_decode.json"

