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

PATH_TRAIN_AUG = 'data/train_augmented_1000_full.csv'
PATH_VAL_CLEAN = 'data/val_clean_50_full.csv'
PATH_TEST = 'data/test.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COLS_INTERNAL = ["target_visc", "target_oxid"]
TARGET_COLS_SUBMISSION = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"
]
DOSE_COL = "mass_norm"
TEMP_COL = "temp"
TIME_COL = "time"
BIO_COL = "biofuel"
CAT_COL = "catalyst"

# --- ПАРАМЕТРЫ АРХИТЕКТУРЫ ---
D_MODEL = 32  
N_HEADS = 4
N_LAYERS = 2  
D_FF = 64
DROPOUT = 0.25
N_SEEDS = 3

# --- ОБУЧЕНИЕ ---
EPOCHS = 1000
LR = 2e-4         
WEIGHT_DECAY = 1e-2
PATIENCE = 150     
BATCH_SIZE = 16
SEED = 42
N_ENSEMBLE_SEEDS = 1

# --- АУГМЕНТАЦИЯ ---
AUG_NOISE_STD = 0.03
AUG_DROP_PROB = 0.05
AUG_DROP_THRESHOLD = 0.1 
AUG_MULTIPLIER = 5
AUG_DOSE_JITTER = 0.015

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

class DOTDataset(Dataset):
    def __init__(self, scenarios, feat_scaler=None, global_scaler=None, target_scaler=None, fit_scalers=False, augment=False):
        self.scenarios = scenarios
        self.augment = augment
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
        return len(self.scenarios) * (AUG_MULTIPLIER if self.augment else 1)

    def __getitem__(self, idx):
        real_idx = idx % len(self.scenarios)
        s = self.scenarios[real_idx]
        comps = s["components_scaled"].copy()
        comp_ids = s["comp_ids"].copy()
        gf = s["global_feats"].copy()

        if self.augment and idx >= len(self.scenarios):
            comps += np.random.randn(*comps.shape).astype(np.float32) * AUG_NOISE_STD
            comps[:, 0] += np.random.randn(comps.shape[0]).astype(np.float32) * AUG_DOSE_JITTER
            gf[:2] += np.random.randn(2).astype(np.float32) * 1.0
            raw_doses = s.get("raw_doses", None)
            if raw_doses is not None and len(comps) > 4:
                keep_mask = np.ones(len(comps), dtype=bool)
                for i in range(len(comps)):
                    if raw_doses[i] < AUG_DROP_THRESHOLD and np.random.rand() < AUG_DROP_PROB:
                        keep_mask[i] = False
                if keep_mask.sum() >= 3:
                    comps = comps[keep_mask]
                    comp_ids = comp_ids[keep_mask]

        gf_scaled = self.global_scaler.transform(gf.reshape(1, -1)).flatten().astype(np.float32)
        if self.has_targets:
            return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled), torch.tensor(s["targets_scaled"])
        return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled)

def collate_fn(batch):
    has_targets = len(batch[0]) == 4
    components, comp_ids = [b[0] for b in batch], [b[1] for b in batch]
    global_feats = torch.stack([b[2] for b in batch])
    targets = torch.stack([b[3] for b in batch]) if has_targets else None

    max_len = max(c.shape[0] for c in components)
    padded = torch.zeros(len(components), max_len, components[0].shape[1])
    padded_ids = torch.zeros(len(components), max_len, dtype=torch.long)
    mask = torch.zeros(len(components), max_len, dtype=torch.bool)
    
    for i, (c, cid) in enumerate(zip(components, comp_ids)):
        n = c.shape[0]
        padded[i, :n], padded_ids[i, :n], mask[i, :n] = c, cid, True

    if targets is not None: return padded, padded_ids, global_feats, mask, targets
    return padded, padded_ids, global_feats, mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k, self.n_heads = d_model // n_heads, n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = [nn.Linear(d_model, d_model) for _ in range(4)]
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
        combined_repr = torch.cat([pooled_flat, global_feats], dim=-1) # LATE FUSION как в коде №1
        return torch.cat([self.head_visc(combined_repr), self.head_oxid(combined_repr)], dim=-1), None

def custom_mae_loss(pred, target):
    loss = F.l1_loss(pred, target, reduction='none')
    weighted_loss = loss * torch.tensor([0.6, 0.4], device=pred.device) 
    return weighted_loss.mean()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, count = 0, 0
    for padded, comp_ids, global_feats, mask, targets in loader:
        padded, comp_ids, global_feats, mask, targets = [x.to(device) for x in [padded, comp_ids, global_feats, mask, targets]]
        optimizer.zero_grad()
        pred, _ = model(padded, comp_ids, global_feats, mask)
        loss = custom_mae_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * padded.shape[0]; count += padded.shape[0]
    return total_loss / count

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, count = 0, 0
    for padded, comp_ids, global_feats, mask, targets in loader:
        padded, comp_ids, global_feats, mask, targets = [x.to(device) for x in [padded, comp_ids, global_feats, mask, targets]]
        pred, _ = model(padded, comp_ids, global_feats, mask)
        loss = custom_mae_loss(pred, targets)
        total_loss += loss.item() * padded.shape[0]; count += padded.shape[0]
    return total_loss / count

train_aug_df = pd.read_csv(PATH_TRAIN_AUG).fillna(0)
val_clean_df = pd.read_csv(PATH_VAL_CLEAN).fillna(0)
test_df = pd.read_csv(PATH_TEST).fillna(0)

rename_map = {'COMP_Ca_cnt': 'cnt_Ca', 'COMP_S_cnt': 'cnt_S', 'COMP_Zn_cnt': 'cnt_Zn', 
              'CHEM_logp': 'logp', 'CHEM_mol_wt': 'mol_wt', 'CHEM_rings': 'rings'}
test_df = test_df.rename(columns=rename_map)

feature_cols = get_feature_columns(train_aug_df)
comp_to_idx = build_component_vocab(train_aug_df, test_df)
n_components = len(comp_to_idx)

train_scenarios = build_scenarios(train_aug_df, comp_to_idx, feature_cols, is_train=True)
val_scenarios = build_scenarios(val_clean_df, comp_to_idx, feature_cols, is_train=True)
test_scenarios = build_scenarios(test_df, comp_to_idx, feature_cols, is_train=False)

final_models, final_scalers = [], []

for seed_i in range(N_ENSEMBLE_SEEDS):
    s = SEED + seed_i * 137
    torch.manual_seed(s)
    np.random.seed(s)

    train_ds = DOTDataset(train_scenarios, fit_scalers=True, augment=True)
    val_ds = DOTDataset(val_scenarios, feat_scaler=train_ds.feat_scaler, global_scaler=train_ds.global_scaler, target_scaler=train_ds.target_scaler, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SetTransformerDOT(feat_dim=len(feature_cols)+1, n_components=n_components, d_model=D_MODEL, 
                              n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, n_seeds=N_SEEDS, dropout=DROPOUT).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    best_loss, best_state, patience_counter = float("inf"), None, 0
    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Seed {seed_i+1}/{N_ENSEMBLE_SEEDS}")
    
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        if val_loss < best_loss:
            best_loss, patience_counter = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: patience_counter += 1

        pbar.set_postfix({'Tr': f"{train_loss:.3f}", 'Val': f"{val_loss:.3f}", 'Best': f"{best_loss:.3f}"})
        if patience_counter >= PATIENCE: break

    model.load_state_dict(best_state)
    final_models.append(model)
    final_scalers.append((train_ds.feat_scaler, train_ds.global_scaler, train_ds.target_scaler))

all_preds = []
for model, (f_sc, g_sc, t_sc) in zip(final_models, final_scalers):
    model.eval()
    test_ds = DOTDataset(test_scenarios, feat_scaler=f_sc, global_scaler=g_sc, target_scaler=t_sc, augment=False)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    fold_preds = []
    with torch.no_grad():
        for padded, comp_ids, global_feats, mask in loader:
            pred, _ = model(padded.to(DEVICE), comp_ids.to(DEVICE), global_feats.to(DEVICE), mask.to(DEVICE))
            fold_preds.append(t_sc.inverse_transform(pred.cpu().numpy()))
    all_preds.append(np.concatenate(fold_preds))

final_test_preds = np.maximum(np.mean(all_preds, axis=0), 0.0)
pred_df = pd.DataFrame({
    "scenario_id": [s["scenario_id"] for s in test_scenarios],
    TARGET_COLS_SUBMISSION[0]: final_test_preds[:, 0],
    TARGET_COLS_SUBMISSION[1]: final_test_preds[:, 1],
})
pred_df.to_csv("predictions_t9.csv", index=False)
print(f"Готово! Сабмит сохранен. Сценариев: {len(pred_df)}")
