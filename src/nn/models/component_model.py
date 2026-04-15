import torch
import torch.nn as nn

class ComponentPropertyEncoder(nn.Module):
    def __init__(self, sbert_dim=768, hidden_dim=32):
        super().__init__()
        # Сжимаем огромный вектор SBERT (768) до чего-то маленького
        # Чтобы не переобучиться на 160 примерах
        self.prop_compressor = nn.Sequential(
            nn.Linear(sbert_dim + 1, hidden_dim), # +1 для числового значения
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, prop_embeddings, prop_values):
        """
        prop_embeddings: (num_properties, 768)
        prop_values: (num_properties, 1)
        """
        # Объединяем вектор названия свойства (SBERT) и его значение
        x = torch.cat([prop_embeddings, prop_values], dim=-1) # (n, 769)
        
        # Прогоняем каждое свойство через нейронку (это наша функция phi)
        prop_feats = self.prop_compressor(x) # (n, hidden_dim)
        
        # Схлопываем все свойства в один вектор компонента (Aggregation)
        # Используем mean, так как количество свойств у компонентов разное
        component_vector = torch.mean(prop_feats, dim=0) # (hidden_dim)
        return component_vector
