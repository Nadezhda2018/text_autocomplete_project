import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutocompleteModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2):
        """
        vocab_size: размер словаря токенизатора
        embedding_dim: размер вектора каждого слова
        hidden_dim: количество нейронов в скрытом слое LSTM
        n_layers: количество слоев LSTM (глубина нейросети)
        """
        super(LSTMAutocompleteModel, self).__init__()
        
        # 1. Слой эмбеддингов (превращает ID слов в смысловые векторы)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Основной слой LSTM
        # batch_first=True позволяет подавать данные в формате (Батч, Длина текста, Вектор)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        # 3. Финальный слой (предсказывает вероятность каждого слова из словаря)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Принимает последовательность токенов и предсказывает следующие.
        """
        # x: [batch_size, seq_len]
        
        embeds = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        
        # Пропускаем через LSTM
        lstm_out, hidden = self.lstm(embeds, hidden) # [batch_size, seq_len, hidden_dim]
        
        # Получаем предсказания для каждого токена
        logits = self.fc(lstm_out) # [batch_size, seq_len, vocab_size]
        
        return logits, hidden

    def generate(self, start_text, tokenizer, max_tokens=10, temperature=1.0, device='cpu'):
        """
        Дополнительный метод для генерации нескольких токенов подряд.
        """
        self.eval() # Режим предсказания (отключаем dropout и т.д.)
        
        # Превращаем начальный текст в тензор
        input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)
        generated = input_ids
        
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Подаем всю текущую последовательность в модель
                logits, hidden = self.forward(generated, hidden)
                
                # Берем предсказание только для последнего токена и делим на температуру
                next_token_logits = logits[:, -1, :] / temperature
                
                # Превращаем в вероятности (Softmax)
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Случайный выбор на основе вероятностей (чтобы текст был живым)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем предсказанное слово к тексту
                generated = torch.cat((generated, next_token), dim=1)
                
                # Если модель выдала токен "конца текста", останавливаемся
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Превращаем ID обратно в понятные слова
        return tokenizer.decode(generated[0], skip_special_tokens=True)