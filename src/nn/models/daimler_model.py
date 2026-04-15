import torch
import torch.nn as nn

class DaimlerOxidationPredictor(nn.Module):
    def __init__(self, component_dim=32, global_dim=4, hidden_dim=64):
        super().__init__()
        
        # Нейронка для обработки каждого компонента в рецептуре
        self.phi = nn.Sequential(
            nn.Linear(component_dim + 1, hidden_dim), # +1 для массовой доли %
            nn.ReLU(),
            nn.Dropout(0.2), # Для борьбы с переобучением
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Финальная голова (Decoder rho)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 2 таргета: Delta Viscosity и Oxidation EOT
        )

    def forward(self, component_vectors, mass_fractions, global_features):
        """
        component_vectors: (batch, n_components, component_dim)
        mass_fractions: (batch, n_components, 1)
        global_features: (batch, global_dim) - T, Time, Catalyst, Biofuel
        """
        # Соединяем вектор компонента с его массовой долей
        # (batch, n, dim+1)
        x = torch.cat([component_vectors, mass_fractions], dim=-1)
        
        # Применяем phi к каждому компоненту (через все батчи и присадки)
        # В PyTorch Linear слой автоматически применится к последней размерности
        e_i = self.phi(x) # (batch, n, hidden_dim)
        
        # Схлопываем рецептуру (Sum pooling) - инвариантность к порядку!
        # Используем сумму, так как массовые доли уже внутри e_i
        e_recipe = torch.sum(e_i, dim=1) # (batch, hidden_dim)
        
        # Добавляем глобальный контекст (Температура, Время и т.д.)
        combined = torch.cat([e_recipe, global_features], dim=-1) # (batch, hidden_dim + 4)
        
        # Предсказываем результат теста
        prediction = self.rho(combined)
        return prediction
