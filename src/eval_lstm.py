import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_model_rouge(model, dataloader, tokenizer, device, limit_samples=100):
    model.eval()
    # Используем rouge1 и rouge2 для коротких текстов
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    r1_scores = []
    r2_scores = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            
            # 1. Готовим вход: первые 3/4 текста
            input_len = int(x.size(1) * 0.75)
            inputs = x[:, :input_len]
            
            # Декодируем оригиналы для сравнения (референсы)
            targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in x]
            
            # 2. Генерируем дополнения
            for j in range(inputs.size(0)):
                prompt = tokenizer.decode(inputs[j], skip_special_tokens=True)
                
                # Дополняем оставшуюся 1/4
                max_gen = x.size(1) - input_len
                generated_full = model.generate(
                    prompt, 
                    tokenizer, 
                    max_tokens=max_gen, 
                    device=device
                )
                
                # 3. Считаем метрики
                score = scorer.score(targets[j], generated_full)
                r1_scores.append(score['rouge1'].fmeasure)
                r2_scores.append(score['rouge2'].fmeasure)
                
                if len(r1_scores) >= limit_samples:
                    break
            if len(r1_scores) >= limit_samples:
                break

    return sum(r1_scores)/len(r1_scores), sum(r2_scores)/len(r2_scores)