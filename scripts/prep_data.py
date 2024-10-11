from transformers import AutoTokenizer # type: ignore
# import dask.dataframe as dd # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import logging
import pandas as pd



def split_datasets(dataset, test_size=0.01):
    train, test = train_test_split(dataset, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=test_size, random_state=42)
    print(f'Splitting is completed. Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}')
    return train, val, test

class TranslateDataset(Dataset):
    """
    A custom Dataset class for loading and tokenizing parallel English and French text data from a CSV file.

    Attributes:
        csv (pd.DataFrame): The input CSV file containing English and French text columns.
        english_values (np.ndarray): Array of English text values from the CSV file.
        french_values (np.ndarray): Array of French text values from the CSV file.
        english_tokenizer (AutoTokenizer): Tokenizer for English text.
        french_tokenizer (AutoTokenizer): Tokenizer for French text.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the tokenized input and output tensors for the given index.
    """
    def __init__(self, csv):
        self.csv = csv
        self.english_values = self.csv['en'].values
        self.french_values = self.csv['fr'].values
        # self.english_values = english_values
        # self.french_values = french_values
        self.english_tokenizer = AutoTokenizer.from_pretrained('/Volumes/daai_ke_team/default/mfg_ai_images/bert_data/bert-base-cased/')
        self.french_tokenizer = AutoTokenizer.from_pretrained('/Volumes/daai_ke_team/default/mfg_ai_images/bert_data/flaubert-base-cased/')

    def __len__(self):
        return len(self.english_values)

    def __getitem__(self, idx):
        english = self.english_values[idx]
        french = self.french_values[idx]
        input_tokenized = self.english_tokenizer(str(english), max_length=35, padding='max_length', truncation=True, return_tensors='pt')
        output_tokenized = self.french_tokenizer(str(french), max_length=35, padding='max_length', truncation=True, return_tensors='pt')
        return input_tokenized['input_ids'].squeeze(0), output_tokenized['input_ids'].squeeze(0)

def get_datasets(csv_path, batch_size, test_size):
    print('Reading CSV file...')
    # csv = dd.read_csv(csv_path).compute()
    csv = pd.read_csv(csv_path)
    # csv = csv[:100000]
    csv['en_len'] = csv['en'].apply(lambda x: len(str(x).split()))
    csv = csv[csv['en_len'] < 30]
    train, val, test = split_datasets(csv, test_size = test_size)
    train_dataset = TranslateDataset(train)
    val_dataset = TranslateDataset(val)
    test_dataset = TranslateDataset(test)
    print('Datasets are created.')
    # Prep the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 54)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 54)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 54)
    print('DataLoaders are created.')
    return train_loader, val_loader, test_loader