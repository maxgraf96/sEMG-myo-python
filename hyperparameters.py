# Batch size, number of epochs, etc.

# BATCH_SIZE = 8
BATCH_SIZE = 256


SEQ_LEN = 151
WAVELET_FEATURE_LEN = 24

DATA_LEN = SEQ_LEN - 1
DATASET_SHIFT_SIZE = 1

LOSS_INDICES = [0, 1, 3, 4, 7, 8, 11, 12]

NUM_EPOCHS = 20000

# Optional data augmentation: adds noise to the input data
# If set to 0, no noise is added
# See DataModule.py for more details
DATASET_UPSCALE_FACTOR = 1