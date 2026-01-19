import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NextTokenDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        """
        csv_path: путь к одному из файлов (train.csv, val.csv или test.csv)
        tokenizer: токенизатор, который мы выбрали (например, GPT2)
        max_length: максимальное количество слов в одном примере
        """
        # Читаем данные. Мы берем первую колонку с текстом.
        df = pd.read_csv(csv_path)
        self.texts = df.iloc[:, 0].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Токенизируем: превращаем текст в тензор с числами
        encoding = self.tokenizer(
            text,
            padding="max_length", # Добиваем нулями до max_length
            truncation=True,      # Обрезаем, если текст длиннее max_length
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0) # Убираем лишнюю размерность
        
        # Логика "Предсказания следующего токена":
        # Если у нас есть токены [A, B, C, D]
        # Вход (x) будет: [A, B, C]
        # Цель (y) будет: [B, C, D] (модель учится предсказывать B по A, C по B и т.д.)
        x = input_ids[:-1]
        y = input_ids[1:]
        
        return x, y