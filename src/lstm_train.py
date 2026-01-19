import torch
import torch.nn as nn
from tqdm import tqdm
from eval_lstm import evaluate_model_rouge

def train_model(model, train_loader, val_loader, tokenizer, epochs=5, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Цикл обучения
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Валидация метрик ROUGE
        r1, r2 = evaluate_model_rouge(model, val_loader, tokenizer, device)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Loss: {total_loss/len(train_loader):.4f}")
        print(f"ROUGE-1: {r1:.4f} | ROUGE-2: {r2:.4f}")
        
        # Примеры автодополнения
        samples = ["i am going to", "today is a", "this movie was"]
        for s in samples:
            res = model.generate(s, tokenizer, max_tokens=5, device=device)
            print(f"Prompt: '{s}' -> Gen: '{res}'")
        print("-" * 30)

    return model