# %% [code]
import warnings
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ПУТИ К ДАННЫМ
PATH_TRAIN_AUG = '/Users/sepilovstepansergeevic/Desktop/Code/neftecode/hack/data/train_augmented_1000_full.csv'
PATH_TEST = '/Users/sepilovstepansergeevic/Desktop/Code/neftecode/hack/data/test.csv'
PATH_CHECKPOINT = "/Users/sepilovstepansergeevic/Desktop/Code/neftecode/hack/data/STransformer_078316.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# КОНСТАНТЫ
TARGET_COLS_INTERNAL = ["target_visc", "target_oxid"]
TARGET_COLS_SUBMISSION = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
]
DOSE_COL = "mass_norm"
D_MODEL = 32  
N_HEADS = 4
N_LAYERS = 2  
D_FF = 64
DROPOUT = 0.25
N_SEEDS = 3
BATCH_SIZE = 16

# %% [code]
def build_component_vocab(train_df, test_df):
    all_comps = sorted(set(train_df["component"].unique()) | set(test_df["component"].unique()))
    comp_to_idx = {c: i + 1 for i, c in enumerate(all_comps)}
    return comp_to_idx

def get_feature_columns(df):
    exclude = ['scenario_id', 'component', 'mass_norm', 'target_visc', 'target_oxid', 'target_visc_log', 'hidden_pct',
               'temp', 'time', 'biofuel', 'catalyst']
    return [c for c in df.columns if c not in exclude]

def build_scenarios(mixture_df, comp_to_idx, feature_cols, is_train=True):
    scenarios = []
    for sid, grp in mixture_df.groupby("scenario_id"):
        comp_features, comp_ids, raw_doses = [], [], []
        for _, row in grp.iterrows():
            comp_name = row["component"]
            dose = row[DOSE_COL]
            raw_doses.append(dose)
            comp_ids.append(comp_to_idx.get(comp_name, 0))
            feats = row[feature_cols].values.astype(np.float32)
            full_vector = np.concatenate([[dose], feats])
            comp_features.append(full_vector)

        first_row = grp.iloc[0]
        global_feats = np.array([first_row['temp'], first_row['time'], first_row['biofuel'], first_row['catalyst']], dtype=np.float32)

        scenario = {
            "components": np.stack(comp_features),
            "comp_ids": np.array(comp_ids, dtype=np.int64),
            "global_feats": global_feats, 
            "raw_doses": np.array(raw_doses, dtype=np.float32),
            "scenario_id": sid
        }
        if is_train:
            targets_raw = np.array([first_row[TARGET_COLS_INTERNAL[0]], first_row[TARGET_COLS_INTERNAL[1]]], dtype=np.float32)
            scenario["targets"] = targets_raw
        scenarios.append(scenario)
    return scenarios

# %% [code]
class DOTDataset(Dataset):
    def __init__(self, scenarios, feat_scaler=None, global_scaler=None, target_scaler=None, fit_scalers=False):
        self.scenarios = scenarios
        self.has_targets = "targets" in scenarios[0]

        if fit_scalers:
            all_feats = np.concatenate([s["components"] for s in scenarios], axis=0)
            all_globals = np.stack([s["global_feats"] for s in scenarios])
            self.feat_scaler = StandardScaler().fit(all_feats)
            self.global_scaler = StandardScaler().fit(all_globals)
            if self.has_targets:
                all_targets = np.stack([s["targets"] for s in scenarios])
                self.target_scaler = StandardScaler().fit(all_targets)
        else:
            self.feat_scaler, self.global_scaler, self.target_scaler = feat_scaler, global_scaler, target_scaler

        for s in self.scenarios:
            s["components_scaled"] = self.feat_scaler.transform(s["components"]).astype(np.float32)
            if self.has_targets and self.target_scaler is not None:
                s["targets_scaled"] = self.target_scaler.transform(s["targets"].reshape(1, -1)).flatten().astype(np.float32)

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        s = self.scenarios[idx]
        comps = s["components_scaled"].copy()
        comp_ids = s["comp_ids"].copy()
        gf = s["global_feats"].copy()
        gf_scaled = self.global_scaler.transform(gf.reshape(1, -1)).flatten().astype(np.float32)
        
        if self.has_targets:
            return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled), torch.tensor(s["targets_scaled"])
        return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled)

def collate_fn(batch):
    has_targets = len(batch[0]) == 4
    components, comp_ids = [b[0] for b in batch], [b[1] for b in batch]
    global_feats = torch.stack([b[2] for b in batch])
    
    max_len = max(c.shape[0] for c in components)
    padded = torch.zeros(len(components), max_len, components[0].shape[1])
    padded_ids = torch.zeros(len(components), max_len, dtype=torch.long)
    mask = torch.zeros(len(components), max_len, dtype=torch.bool)
    
    for i, (c, cid) in enumerate(zip(components, comp_ids)):
        n = c.shape[0]
        padded[i, :n], padded_ids[i, :n], mask[i, :n] = c, cid, True

    if has_targets:
        targets = torch.stack([b[3] for b in batch])
        return padded, padded_ids, global_feats, mask, targets
    return padded, padded_ids, global_feats, mask

# %% [code]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k, self.n_heads = d_model // n_heads, n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        B, n_q, _ = Q.shape
        _, n_kv, _ = K.shape
        q = self.W_q(Q).view(B, n_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, n_kv, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, n_kv, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, n_q, -1)
        return self.W_o(out), attn_weights

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, X, mask=None):
        attn_out, _ = self.mha(X, X, X, mask)
        X = self.norm1(X + attn_out)
        return self.norm2(X + self.ffn(X)), None

class PoolingByMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.01)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z, mask=None):
        out, _ = self.mha(self.seeds.expand(Z.shape[0], -1, -1), Z, Z, mask)
        return self.norm(self.seeds.expand(Z.shape[0], -1, -1) + out), None

class SetTransformerDOT(nn.Module):
    def __init__(self, feat_dim, n_components, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_seeds=3, dropout=0.25, comp_embed_dim=8):
        super().__init__()
        self.comp_embedding = nn.Embedding(n_components + 1, comp_embed_dim, padding_idx=0)
        self.input_proj = nn.Sequential(nn.Linear(feat_dim + comp_embed_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.sab_layers = nn.ModuleList([SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.pma = PoolingByMultiHeadAttention(d_model, n_heads, n_seeds, dropout)
        
        head_in_dim = (d_model * n_seeds) + 4
        self.head_visc = nn.Sequential(nn.Linear(head_in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.head_oxid = nn.Sequential(nn.Linear(head_in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x, comp_ids, global_feats, mask=None):
        h = self.input_proj(torch.cat([x, self.comp_embedding(comp_ids)], dim=-1))
        if mask is not None: h = h * mask.unsqueeze(-1).float()
        for sab in self.sab_layers:
            h, _ = sab(h, mask)
            if mask is not None: h = h * mask.unsqueeze(-1).float()
        pooled, _ = self.pma(h, mask)
        pooled_flat = pooled.view(pooled.shape[0], -1)
        combined_repr = torch.cat([pooled_flat, global_feats], dim=-1)
        return torch.cat([self.head_visc(combined_repr), self.head_oxid(combined_repr)], dim=-1), None

# %% [code]
train_aug_df = pd.read_csv(PATH_TRAIN_AUG).fillna(0)
test_df = pd.read_csv(PATH_TEST).fillna(0)

# Переименование колонок в тесте (если они отличаются от трейна)
rename_map = {'COMP_Ca_cnt': 'cnt_Ca', 'COMP_S_cnt': 'cnt_S', 'COMP_Zn_cnt': 'cnt_Zn', 
              'CHEM_logp': 'logp', 'CHEM_mol_wt': 'mol_wt', 'CHEM_rings': 'rings'}
test_df = test_df.rename(columns=rename_map)

# Получаем признаки и словарь компонентов
feature_cols = get_feature_columns(train_aug_df)
comp_to_idx = build_component_vocab(train_aug_df, test_df)
n_components = len(comp_to_idx)

# Собираем сценарии
train_scenarios = build_scenarios(train_aug_df, comp_to_idx, feature_cols, is_train=True)
test_scenarios = build_scenarios(test_df, comp_to_idx, feature_cols, is_train=False)

# Обучаем скелеры на трейне (важно для корректного предсказания)
train_ds = DOTDataset(train_scenarios, fit_scalers=True)
f_sc, g_sc, t_sc = train_ds.feat_scaler, train_ds.global_scaler, train_ds.target_scaler

# %% [code]
# Загружаем всё содержимое сразу
checkpoint = torch.load(PATH_CHECKPOINT, map_location=DEVICE, weights_only=False)

# Извлекаем словари и параметры
comp_to_idx = checkpoint['comp_to_idx']
feature_cols = checkpoint['feature_cols']
model_config = checkpoint['model_config']

# Извлекаем предобученные скалеры
f_sc = checkpoint['feat_scaler']
g_sc = checkpoint['global_scaler']
t_sc = checkpoint['target_scaler']

print(f"Чекпоинт загружен. Найдено признаков: {len(feature_cols)}")

# %% [code]
# Создаем модель, используя конфиг из чекпоинта
model = SetTransformerDOT(
    feat_dim=model_config['feat_dim'],
    n_components=model_config['n_components'],
    d_model=model_config['d_model'],
    n_heads=model_config['n_heads'],
    n_layers=model_config['n_layers'],
    d_ff=model_config['d_ff'],
    n_seeds=model_config['n_seeds'],
    dropout=model_config['dropout']
).to(DEVICE)

# Загружаем веса
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Архитектура модели воссоздана, веса загружены.")

# %% [code]
# Загружаем тестовый файл
test_df = pd.read_csv(PATH_TEST).fillna(0)

# Маппинг колонок (если в тесте они называются иначе, как в твоем исходном коде)
rename_map = {'COMP_Ca_cnt': 'cnt_Ca', 'COMP_S_cnt': 'cnt_S', 'COMP_Zn_cnt': 'cnt_Zn', 
              'CHEM_logp': 'logp', 'CHEM_mol_wt': 'mol_wt', 'CHEM_rings': 'rings'}
test_df = test_df.rename(columns=rename_map)

# Собираем сценарии, используя feature_cols и comp_to_idx из чекпоинта
test_scenarios = build_scenarios(test_df, comp_to_idx, feature_cols, is_train=False)

# Создаем датасет с загруженными скалерами
test_ds = DOTDataset(test_scenarios, feat_scaler=f_sc, global_scaler=g_sc, target_scaler=t_sc)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        padded, comp_ids, global_feats, mask = [x.to(DEVICE) for x in batch]
        pred, _ = model(padded, comp_ids, global_feats, mask)
        
        # Инвертируем нормализацию таргетов
        pred_original_scale = t_sc.inverse_transform(pred.cpu().numpy())
        all_preds.append(pred_original_scale)

final_test_preds = np.concatenate(all_preds)
final_test_preds = np.maximum(final_test_preds, 0.0)

pred_df = pd.DataFrame({
    "scenario_id": [s["scenario_id"] for s in test_scenarios],
    TARGET_COLS_SUBMISSION[0]: final_test_preds[:, 0],
    TARGET_COLS_SUBMISSION[1]: final_test_preds[:, 1],
})

pred_df.to_csv("submission_final.csv", index=False)
print(f"Готово! Предсказания сохранены. Сценариев обработано: {len(pred_df)}")
