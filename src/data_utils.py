import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Очистка текста твитов."""
    if not isinstance(text, str): 
        return ""
    
    text = text.lower()
    # Удаляем ссылки
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Удаляем упоминания @user
    text = re.sub(r'@\w+', '', text)
    # Оставляем только буквы (английские и русские) и пробелы
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(input_path, output_dir='data'):
    """
    Полный цикл обработки согласно структуре проекта.
    """
    print(f"Читаем сырые данные: {input_path}")
    
    # Твои данные в .txt, читаем их построчно
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    df = pd.DataFrame(lines, columns=['raw_text'])
    
    # 1. Чистим данные
    print("Очистка текстов...")
    df['cleaned_text'] = df['raw_text'].apply(clean_text)
    
    # Удаляем пустые строки
    df = df[df['cleaned_text'] != ""].copy()
    
    # 2. Сохраняем "очищенный" датасет (dataset_processed.csv)
    processed_path = os.path.join(output_dir, 'dataset_processed.csv')
    df[['cleaned_text']].to_csv(processed_path, index=False)
    print(f"Очищенный датасет сохранен в {processed_path}")
    
    # 3. Разбиваем на выборки (80 / 10 / 10)
    print("Разбиение на выборки...")
    train_df, temp_df = train_test_split(df['cleaned_text'], test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # 4. Сохраняем финальные части
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("Все файлы (train/val/test) успешно созданы в папке data/")