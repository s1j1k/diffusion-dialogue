# Might need to install this gdown to download the file from the google drive
# pip install gdown

import json
import os
import gdown

def download_file_from_google_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(e)
                    continue
    return data

def prepare_datasets(train_id, valid_id, test_id, dest_dir='datasets'):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Paths for the datasets
    train_path = os.path.join(dest_dir, 'train_full.jsonl')
    valid_path = os.path.join(dest_dir, 'valid_full.jsonl')
    test_path = os.path.join(dest_dir, 'test_full.jsonl')
    
    # Download datasets if they do not exist
    if not os.path.exists(train_path):
        download_file_from_google_drive(train_id, train_path)
    if not os.path.exists(valid_path):
        download_file_from_google_drive(valid_id, valid_path)
    if not os.path.exists(test_path):
        download_file_from_google_drive(test_id, test_path)

    # Load datasets
    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)
    test_data = load_jsonl(test_path)
    
    return train_data, valid_data, test_data

# Define the Google Drive file IDs
TRAIN_FILE_ID = '1yTLSDRdhMokjBixsp1V359sSQf4T7eiC'
VALID_FILE_ID = '1VLYAh78lTX6yYJjqke6lSbqe95MVXFuF'
TEST_FILE_ID = '1Utn3a9NRzO5W4ZZe6tyrMyKWuGdfIiwF'

if __name__ == "__main__":
    train_data, valid_data, test_data = prepare_datasets(TRAIN_FILE_ID, VALID_FILE_ID, TEST_FILE_ID)
    print(f"Loaded {len(train_data)} training records.")
    print(f"Loaded {len(valid_data)} validation records.")
    print(f"Loaded {len(test_data)} test records.")
