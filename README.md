# diffusion-dialogue

## References
DiffuSeq
- refer `train.py`

- used example dataset

Use the data_loader.py to load the dataset from google drive
'''
from data_loader import prepare_datasets, TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID
train_data, valid_data, test_data = prepare_datasets(TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID)
'''

