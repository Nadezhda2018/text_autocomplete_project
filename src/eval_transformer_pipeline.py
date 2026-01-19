import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_transformer_rouge(model_name="distilgpt2", test_texts=None, device="cpu"):
    # 1. Загружаем модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # У GPT-2 нет pad_token
    
    # Создаем пайплайн для генерации текста
    gen_pipeline = pipeline(
        "text-generation", 
        model=model_name, 
        tokenizer=tokenizer, 
        device=0 if device == "cuda" else -1
    )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    r1_scores, r2_scores = [], []
    
    # Берем небольшую выборку для теста (например, 100 текстов)
    sample_texts = test_texts[:100] if test_texts else []
    
    print(f"Оценка Трансформера {model_name}...")
    for text in tqdm(sample_texts):
        # Делим текст: 3/4 вход, 1/4 таргет
        words = text.split()
        input_len = int(len(words) * 0.75)
        if input_len == 0: continue
        
        prompt = " ".join(words[:input_len])
        target = text
        
        # Генерация (подбираем параметры: top_k, temperature)
        # max_new_tokens ставим примерно на 1/4 длины
        output = gen_pipeline(
            prompt, 
            max_new_tokens=max(5, len(words) - input_len), 
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.8,
            top_k=50,
            do_sample=True
        )
        
        generated_text = output[0]['generated_text']
        
        # Считаем ROUGE
        score = scorer.score(target, generated_text)
        r1_scores.append(score['rouge1'].fmeasure)
        r2_scores.append(score['rouge2'].fmeasure)
        
    avg_r1 = sum(r1_scores)/len(r1_scores) if r1_scores else 0
    avg_r2 = sum(r2_scores)/len(r2_scores) if r2_scores else 0
    
    return avg_r1, avg_r2, generated_text # Возвращаем еще и пример последнего текста