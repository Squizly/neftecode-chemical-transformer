import math
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

TARGET_COLS = [
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]
DOSE_COL = "Массовая доля, %"
TEMP_COL = "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C"
TIME_COL = "Время испытания | - Daimler Oxidation Test (DOT), ч"
BIO_COL = "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс"
CAT_COL = "Дозировка катализатора, категория"

# --- Architecture ---
D_MODEL = 48
N_HEADS = 4
N_LAYERS = 2
D_FF = 96
DROPOUT = 0.20
N_SEEDS = 3

# --- Training ---
N_FOLDS = 5
EPOCHS = 1000
LR = 2e-4
WEIGHT_DECAY = 5e-3
PATIENCE = 120
BATCH_SIZE = 16
SEED = 42
N_ENSEMBLE_SEEDS = 5  # train multiple models per fold

# --- Augmentation ---
AUG_NOISE_STD = 0.02
AUG_DROP_PROB = 0.05
AUG_DROP_THRESHOLD = 10.0
AUG_MULTIPLIER = 5
AUG_DOSE_JITTER = 0.015

# --- Ensemble weights ---
RIDGE_WEIGHT = 0.15  
ST_WEIGHT = 0.85


def log_transform(y):
    return np.sign(y) * np.log1p(np.abs(y))

def log_inverse(y):
    return np.sign(y) * np.expm1(np.abs(y))


def extract_component_type(name: str) -> str:
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name


def load_properties(path: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["Компонент", "Партия", "Показатель", "Единица", "Значение"]
    df["Значение"] = pd.to_numeric(df["Значение"], errors="coerce")
    props = {}
    for (comp, batch), grp in df.groupby(["Компонент", "Партия"]):
        d = {}
        for _, row in grp.iterrows():
            if pd.notna(row["Значение"]):
                d[row["Показатель"]] = row["Значение"]
        props[(comp, str(batch))] = d
    return props


def get_component_properties(comp, batch, props_dict, prop_names):
    key = (comp, str(batch))
    typical_key = (comp, "typical")
    batch_props = props_dict.get(key, {})
    typical_props = props_dict.get(typical_key, {})
    values, mask = [], []
    for pname in prop_names:
        if pname in batch_props:
            values.append(batch_props[pname])
            mask.append(1.0)
        elif pname in typical_props:
            values.append(typical_props[pname])
            mask.append(1.0)
        else:
            values.append(0.0)
            mask.append(0.0)
    return np.array(values, dtype=np.float32), np.array(mask, dtype=np.float32)


def select_numeric_properties(props_dict, min_coverage=0.03):
    all_keys = set()
    for d in props_dict.values():
        all_keys.update(d.keys())
    total = len(props_dict)
    selected = []
    for key in sorted(all_keys):
        count = sum(1 for d in props_dict.values() if key in d)
        if count / total >= min_coverage:
            selected.append(key)
    return selected


def build_component_vocab(train_df, test_df):
    """Build component name -> index mapping for embeddings."""
    all_comps = sorted(set(train_df["Компонент"].unique()) | set(test_df["Компонент"].unique()))
    comp_to_idx = {c: i + 1 for i, c in enumerate(all_comps)}  # 0 reserved for unknown
    return comp_to_idx


def build_scenarios(mixture_df, props_dict, prop_names, comp_type_to_idx,
                    comp_to_idx, is_train=True):
    dot_condition_cols = [TEMP_COL, TIME_COL, BIO_COL, CAT_COL]
    n_types = len(comp_type_to_idx)
    scenarios = []

    for sid, grp in mixture_df.groupby("scenario_id"):
        comp_features = []
        comp_ids = []
        raw_doses = []

        for _, row in grp.iterrows():
            comp_name = row["Компонент"]
            batch = str(row["Наименование партии"]) if pd.notna(row["Наименование партии"]) else ""
            dose = row[DOSE_COL]
            raw_doses.append(dose)

            # Component ID for embedding
            comp_ids.append(comp_to_idx.get(comp_name, 0))

            # Component type one-hot
            ctype = extract_component_type(comp_name)
            type_vec = np.zeros(n_types, dtype=np.float32)
            if ctype in comp_type_to_idx:
                type_vec[comp_type_to_idx[ctype]] = 1.0

            # Properties
            prop_vals, prop_mask = get_component_properties(
                comp_name, batch, props_dict, prop_names
            )

            # DOT conditions
            dot_conds = np.array([row[c] for c in dot_condition_cols], dtype=np.float32)

            # Dose interactions with conditions
            dose_x_temp = dose * dot_conds[0]
            dose_x_time = dose * dot_conds[1]
            dose_x_bio = dose * dot_conds[2]

            feat = np.concatenate([
                [dose], type_vec, prop_vals, prop_mask, dot_conds,
                [dose_x_temp, dose_x_time, dose_x_bio],
            ])
            comp_features.append(feat)

        scenario = {
            "components": np.stack(comp_features),
            "comp_ids": np.array(comp_ids, dtype=np.int64),
            "raw_doses": np.array(raw_doses, dtype=np.float32),
            "scenario_id": sid,
        }

        # DOT condition key for stratification
        first_row = grp.iloc[0]
        scenario["strat_key"] = f"{int(first_row[TEMP_COL])}_{int(first_row[TIME_COL])}_{int(first_row[BIO_COL])}"

        if is_train:
            scenario["targets_orig"] = np.array([
                grp[TARGET_COLS[0]].iloc[0],
                grp[TARGET_COLS[1]].iloc[0],
            ], dtype=np.float32)
            scenario["targets"] = np.array([
                log_transform(grp[TARGET_COLS[0]].iloc[0]),
                grp[TARGET_COLS[1]].iloc[0],
            ], dtype=np.float32)
        scenarios.append(scenario)

    return scenarios


def build_handcrafted_features(scenarios, comp_type_to_idx, prop_names):
    n_types = len(comp_type_to_idx)
    feat_list = []

    for s in scenarios:
        comps = s["components"]
        n_comp = comps.shape[0]
        doses = comps[:, 0]
        type_onehots = comps[:, 1:1 + n_types]
        prop_start = 1 + n_types
        prop_end = prop_start + len(prop_names)
        properties = comps[:, prop_start:prop_end]
        dot_conds = comps[0, prop_end + len(prop_names):prop_end + len(prop_names) + 4]
        # Actually, let's re-extract DOT from the raw data layout
        # Layout: [dose(1), type(n_types), props(n_props), prop_mask(n_props), dot(4), interactions(3)]
        mask_end = prop_end + len(prop_names)
        dot_conds = comps[0, mask_end:mask_end + 4]

        type_dose_sum = np.zeros(n_types, dtype=np.float32)
        type_dose_max = np.zeros(n_types, dtype=np.float32)
        type_dose_min = np.full(n_types, 999.0, dtype=np.float32)
        type_count = np.zeros(n_types, dtype=np.float32)
        for i in range(n_comp):
            t_idx = np.argmax(type_onehots[i])
            if type_onehots[i, t_idx] > 0:
                type_dose_sum[t_idx] += doses[i]
                type_dose_max[t_idx] = max(type_dose_max[t_idx], doses[i])
                type_dose_min[t_idx] = min(type_dose_min[t_idx], doses[i])
                type_count[t_idx] += 1
        type_dose_min[type_dose_min > 900] = 0.0

        total_dose = doses.sum() + 1e-8
        weighted_props = (properties * doses[:, None]).sum(axis=0) / total_dose
        
        # Property statistics
        prop_mask = comps[:, prop_end:mask_end]
        prop_coverage = prop_mask.mean(axis=0)  # average coverage per property

        # Pairwise type interactions
        pairwise = []
        for i in range(n_types):
            for j in range(i + 1, n_types):
                pairwise.append(type_dose_sum[i] * type_dose_sum[j])
        pairwise = np.array(pairwise, dtype=np.float32)

        # DOT condition interactions with type doses
        temp, time, bio, cat = dot_conds
        dot_x_type = []
        for i in range(n_types):
            dot_x_type.extend([
                type_dose_sum[i] * temp,
                type_dose_sum[i] * time,
                type_dose_sum[i] * bio,
            ])
        dot_x_type = np.array(dot_x_type, dtype=np.float32)

        # Condition interactions
        cond_interactions = np.array([
            temp * time,
            temp * bio,
            time * bio,
            temp * time * bio,
            cat * temp,
            cat * bio,
        ], dtype=np.float32)

        general = np.array([
            n_comp,
            (type_count > 0).sum(),
            doses.sum(),
            doses.mean(),
            doses.std(),
            doses.max(),
            doses.min(),
            np.log1p(doses.sum()),
        ], dtype=np.float32)

        feat = np.concatenate([
            type_dose_sum,
            type_dose_max,
            type_dose_min,
            type_count,
            weighted_props,
            prop_coverage,
            pairwise,
            general,
            dot_conds,
            cond_interactions,
            dot_x_type,
        ])
        feat_list.append(feat)

    return np.stack(feat_list)


class DOTDataset(Dataset):
    def __init__(self, scenarios, feat_scaler=None, target_scaler=None,
                 fit_scalers=False, augment=False):
        self.scenarios = scenarios
        self.augment = augment
        self.has_targets = "targets" in scenarios[0]

        all_feats = np.concatenate([s["components"] for s in scenarios], axis=0)

        if fit_scalers:
            self.feat_scaler = RobustScaler().fit(all_feats)
            if self.has_targets:
                all_targets = np.stack([s["targets"] for s in scenarios])
                self.target_scaler = RobustScaler().fit(all_targets)
            else:
                self.target_scaler = None
        else:
            self.feat_scaler = feat_scaler
            self.target_scaler = target_scaler

        for s in self.scenarios:
            s["components_scaled"] = self.feat_scaler.transform(
                s["components"]
            ).astype(np.float32)
            if self.has_targets and self.target_scaler is not None:
                s["targets_scaled"] = self.target_scaler.transform(
                    s["targets"].reshape(1, -1)
                ).flatten().astype(np.float32)

    def __len__(self):
        return len(self.scenarios) * (AUG_MULTIPLIER if self.augment else 1)

    def __getitem__(self, idx):
        real_idx = idx % len(self.scenarios)
        s = self.scenarios[real_idx]
        comps = s["components_scaled"].copy()
        comp_ids = s["comp_ids"].copy()

        if self.augment and idx >= len(self.scenarios):
            noise = np.random.randn(*comps.shape).astype(np.float32) * AUG_NOISE_STD
            comps = comps + noise

            # Dose jitter (first column)
            dose_noise = np.random.randn(comps.shape[0]).astype(np.float32) * AUG_DOSE_JITTER
            comps[:, 0] += dose_noise

            raw_doses = s.get("raw_doses", None)
            if raw_doses is not None and len(comps) > 4:
                keep_mask = np.ones(len(comps), dtype=bool)
                for i in range(len(comps)):
                    if raw_doses[i] < AUG_DROP_THRESHOLD and np.random.rand() < AUG_DROP_PROB:
                        keep_mask[i] = False
                if keep_mask.sum() >= 3:
                    comps = comps[keep_mask]
                    comp_ids = comp_ids[keep_mask]

        comps = torch.tensor(comps, dtype=torch.float32)
        comp_ids = torch.tensor(comp_ids, dtype=torch.long)
        if self.has_targets:
            targets = torch.tensor(s["targets_scaled"], dtype=torch.float32)
            return comps, comp_ids, targets
        return comps, comp_ids


def collate_fn(batch):
    has_targets = len(batch[0]) == 3
    if has_targets:
        components = [b[0] for b in batch]
        comp_ids = [b[1] for b in batch]
        targets = torch.stack([b[2] for b in batch])
    else:
        components = [b[0] for b in batch]
        comp_ids = [b[1] for b in batch]
        targets = None

    max_len = max(c.shape[0] for c in components)
    feat_dim = components[0].shape[1]
    padded = torch.zeros(len(components), max_len, feat_dim)
    padded_ids = torch.zeros(len(components), max_len, dtype=torch.long)
    mask = torch.zeros(len(components), max_len, dtype=torch.bool)
    for i, (c, cid) in enumerate(zip(components, comp_ids)):
        n = c.shape[0]
        padded[i, :n] = c
        padded_ids[i, :n] = cid
        mask[i, :n] = True

    if targets is not None:
        return padded, padded_ids, mask, targets
    return padded, padded_ids, mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
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
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, n_q, -1)
        out = self.W_o(out)
        return out, attn_weights


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, mask=None):
        attn_out, attn_weights = self.mha(X, X, X, mask)
        X = self.norm1(X + attn_out)
        X = self.norm2(X + self.ffn(X))
        return X, attn_weights


class PoolingByMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.01)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Z, mask=None):
        B = Z.shape[0]
        S = self.seeds.expand(B, -1, -1)
        out, attn_weights = self.mha(S, Z, Z, mask)
        out = self.norm(S + out)
        return out, attn_weights


class SetTransformerDOT(nn.Module):
    def __init__(self, feat_dim, n_components, d_model=48, n_heads=4,
                 n_layers=2, d_ff=96, n_seeds=3, dropout=0.25,
                 comp_embed_dim=8):
        super().__init__()
        self.comp_embed_dim = comp_embed_dim
        self.comp_embedding = nn.Embedding(n_components + 1, comp_embed_dim, padding_idx=0)

        total_input_dim = feat_dim + comp_embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pma = PoolingByMultiHeadAttention(d_model, n_heads, n_seeds, dropout)

        # Dual head: separate heads for viscosity and oxidation
        self.head_visc = nn.Sequential(
            nn.Linear(d_model * n_seeds, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.head_oxid = nn.Sequential(
            nn.Linear(d_model * n_seeds, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, comp_ids, mask=None):
        # Concatenate component embeddings with features
        comp_emb = self.comp_embedding(comp_ids)
        h = torch.cat([x, comp_emb], dim=-1)

        h = self.input_proj(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        all_attn = []
        for sab in self.sab_layers:
            h, attn_w = sab(h, mask)
            all_attn.append(attn_w)
            if mask is not None:
                h = h * mask.unsqueeze(-1).float()
        pooled, pma_attn = self.pma(h, mask)
        all_attn.append(pma_attn)
        pooled_flat = pooled.view(pooled.shape[0], -1)

        pred_visc = self.head_visc(pooled_flat)
        pred_oxid = self.head_oxid(pooled_flat)
        pred = torch.cat([pred_visc, pred_oxid], dim=-1)
        return pred, all_attn


def weighted_huber_loss(pred, target, delta=1.0):
    """Huber loss with equal weighting per target column."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    huber = torch.where(abs_diff <= delta,
                        0.5 * diff ** 2,
                        delta * (abs_diff - 0.5 * delta))
    # Average per target, then average across targets (equal weight)
    return huber.mean(dim=0).mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    for batch in loader:
        if len(batch) == 4:
            padded, comp_ids, mask, targets = batch
        else:
            padded, comp_ids, mask = batch
            continue
        padded = padded.to(device)
        comp_ids = comp_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        pred, _ = model(padded, comp_ids, mask)
        loss = weighted_huber_loss(pred, targets, delta=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * padded.shape[0]
        count += padded.shape[0]
    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0
    preds, trues = [], []
    for batch in loader:
        if len(batch) == 4:
            padded, comp_ids, mask, targets = batch
        else:
            padded, comp_ids, mask = batch
            continue
        padded = padded.to(device)
        comp_ids = comp_ids.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        pred, _ = model(padded, comp_ids, mask)
        loss = weighted_huber_loss(pred, targets, delta=1.0)
        total_loss += loss.item() * padded.shape[0]
        count += padded.shape[0]
        preds.append(pred.cpu())
        trues.append(targets.cpu())
    return total_loss / count, torch.cat(preds), torch.cat(trues)


def get_strat_labels(scenarios):
    """Create stratification labels from condition groups."""
    strat_keys = [s["strat_key"] for s in scenarios]
    unique_keys = sorted(set(strat_keys))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    return np.array([key_to_idx[k] for k in strat_keys])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("Set Transformer + Ridge Hybrid")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    print("\n[1/6] Загрузка данных...")
    train_df = pd.read_csv(DATA_DIR / "daimler_mixtures_train.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(DATA_DIR / "daimler_mixtures_test.csv", encoding="utf-8-sig")
    props_dict = load_properties(DATA_DIR / "daimler_component_properties.csv")

    all_comps = pd.concat([train_df["Компонент"], test_df["Компонент"]])
    comp_types = sorted(set(all_comps.map(extract_component_type)))
    comp_type_to_idx = {t: i for i, t in enumerate(comp_types)}
    prop_names = select_numeric_properties(props_dict, min_coverage=0.03)
    comp_to_idx = build_component_vocab(train_df, test_df)
    n_components = len(comp_to_idx)

    print(f"  Типов компонентов: {len(comp_types)}")
    print(f"  Числовых свойств: {len(prop_names)}")
    print(f"  Уникальных компонентов (для embeddings): {n_components}")

    print("\n[2/6] Построение признаков...")
    train_scenarios = build_scenarios(
        train_df, props_dict, prop_names, comp_type_to_idx, comp_to_idx, is_train=True
    )
    test_scenarios = build_scenarios(
        test_df, props_dict, prop_names, comp_type_to_idx, comp_to_idx, is_train=False
    )
    feat_dim = train_scenarios[0]["components"].shape[1]
    print(f"  Сценариев: train={len(train_scenarios)}, test={len(test_scenarios)}")
    print(f"  Размерность (без embeddings): {feat_dim}")

    print("\n[3/6] Hand-crafted фичи для Ridge...")
    hc_train = build_handcrafted_features(train_scenarios, comp_type_to_idx, prop_names)
    hc_test = build_handcrafted_features(test_scenarios, comp_type_to_idx, prop_names)
    print(f"  Размерность hand-crafted: {hc_train.shape[1]}")

    y_train_ridge = np.stack([s["targets"] for s in train_scenarios])

    # Stratification labels
    strat_labels = get_strat_labels(train_scenarios)
    print(f"  Стратификация: {len(set(strat_labels))} групп")
    for key in sorted(set([s['strat_key'] for s in train_scenarios])):
        count = sum(1 for s in train_scenarios if s['strat_key'] == key)
        print(f"    {key}: {count}")

    print(f"\n[4/6] {N_FOLDS}-fold Stratified кросс-валидация x {N_ENSEMBLE_SEEDS} seeds...")

    # Use StratifiedKFold with condition-based stratification
    # For groups too small (< N_FOLDS), merge them
    label_counts = pd.Series(strat_labels).value_counts()
    label_map = {}
    for label, count in label_counts.items():
        if count < N_FOLDS:
            # Merge small groups: map to nearest large group
            label_map[label] = label_counts.index[0]  # merge into largest group
        else:
            label_map[label] = label
    merged_labels = np.array([label_map[l] for l in strat_labels])

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_st_models = []
    fold_ridge_models = []
    fold_scalers = []
    cv_scores_st = []
    cv_scores_ridge = []
    cv_scores_hybrid = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_scenarios)), merged_labels)):
        print(f"\n  ── Fold {fold + 1}/{N_FOLDS} ──")
        print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")

        # --- Ridge (separate per target) ---
        hc_tr, hc_val = hc_train[train_idx], hc_train[val_idx]
        y_tr, y_val = y_train_ridge[train_idx], y_train_ridge[val_idx]
        y_val_orig = np.stack([train_scenarios[i]["targets_orig"] for i in val_idx])

        hc_scaler = RobustScaler().fit(hc_tr)
        hc_tr_s = hc_scaler.transform(hc_tr)
        hc_val_s = hc_scaler.transform(hc_val)

        # Separate scalers and models per target
        y_scaler_visc = RobustScaler().fit(y_tr[:, 0:1])
        y_scaler_oxid = RobustScaler().fit(y_tr[:, 1:2])
        y_tr_visc_s = y_scaler_visc.transform(y_tr[:, 0:1]).ravel()
        y_tr_oxid_s = y_scaler_oxid.transform(y_tr[:, 1:2]).ravel()

        # Visc: try HuberRegressor (robust to outliers) + Ridge
        best_visc_mae = float('inf')
        best_visc_model = None
        for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
            m = Ridge(alpha=alpha)
            m.fit(hc_tr_s, y_tr_visc_s)
            p = y_scaler_visc.inverse_transform(m.predict(hc_val_s).reshape(-1, 1)).ravel()
            p_orig = log_inverse(p)
            mae = np.abs(p_orig - y_val_orig[:, 0]).mean()
            if mae < best_visc_mae:
                best_visc_mae = mae
                best_visc_model = m
        # Also try Huber
        for eps in [1.1, 1.35, 1.5, 2.0]:
            try:
                m = HuberRegressor(epsilon=eps, max_iter=500, alpha=1.0)
                m.fit(hc_tr_s, y_tr_visc_s)
                p = y_scaler_visc.inverse_transform(m.predict(hc_val_s).reshape(-1, 1)).ravel()
                p_orig = log_inverse(p)
                mae = np.abs(p_orig - y_val_orig[:, 0]).mean()
                if mae < best_visc_mae:
                    best_visc_mae = mae
                    best_visc_model = m
            except Exception:
                pass

        # Oxid: Ridge
        best_oxid_mae = float('inf')
        best_oxid_model = None
        for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
            m = Ridge(alpha=alpha)
            m.fit(hc_tr_s, y_tr_oxid_s)
            p = y_scaler_oxid.inverse_transform(m.predict(hc_val_s).reshape(-1, 1)).ravel()
            mae = np.abs(p - y_val_orig[:, 1]).mean()
            if mae < best_oxid_mae:
                best_oxid_mae = mae
                best_oxid_model = m

        # Reconstruct ridge predictions
        ridge_pred_visc = y_scaler_visc.inverse_transform(
            best_visc_model.predict(hc_val_s).reshape(-1, 1)).ravel()
        ridge_pred_oxid = y_scaler_oxid.inverse_transform(
            best_oxid_model.predict(hc_val_s).reshape(-1, 1)).ravel()
        ridge_pred_orig = np.column_stack([log_inverse(ridge_pred_visc), ridge_pred_oxid])

        mae_ridge = np.abs(ridge_pred_orig - y_val_orig).mean(axis=0)
        print(f"    Ridge  MAE: visc={mae_ridge[0]:.2f}, oxid={mae_ridge[1]:.2f}")
        cv_scores_ridge.append(mae_ridge)

        fold_ridge_models.append((best_visc_model, best_oxid_model,
                                  hc_scaler, y_scaler_visc, y_scaler_oxid))

        # --- Set Transformer (multiple seeds) ---
        fold_seed_models = []
        fold_seed_scalers = []
        fold_st_preds = []

        for seed_i in range(N_ENSEMBLE_SEEDS):
            s = SEED * 7 + fold * 1000 + seed_i * 137 + 13
            torch.manual_seed(s)
            np.random.seed(s)

            fold_train = [copy.deepcopy(train_scenarios[i]) for i in train_idx]
            fold_val = [copy.deepcopy(train_scenarios[i]) for i in val_idx]

            train_ds = DOTDataset(fold_train, fit_scalers=True, augment=True)
            val_ds = DOTDataset(fold_val, feat_scaler=train_ds.feat_scaler,
                                target_scaler=train_ds.target_scaler, augment=False)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                    shuffle=False, collate_fn=collate_fn)

            model = SetTransformerDOT(
                feat_dim=feat_dim, n_components=n_components,
                d_model=D_MODEL, n_heads=N_HEADS,
                n_layers=N_LAYERS, d_ff=D_FF, n_seeds=N_SEEDS, dropout=DROPOUT,
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=100, T_mult=2
            )

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
                val_loss, val_preds, val_trues = evaluate(model, val_loader, DEVICE)
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 200 == 0:
                    print(f"      Seed {seed_i} Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

                if patience_counter >= PATIENCE:
                    print(f"      Seed {seed_i} Early stop @ epoch {epoch}")
                    break

            model.load_state_dict(best_state)
            model.eval()

            _, val_preds, val_trues = evaluate(model, val_loader, DEVICE)
            st_pred = train_ds.target_scaler.inverse_transform(val_preds.numpy())
            st_pred_orig = st_pred.copy()
            st_pred_orig[:, 0] = log_inverse(st_pred[:, 0])
            fold_st_preds.append(st_pred_orig)

            fold_seed_models.append(model)
            fold_seed_scalers.append((train_ds.feat_scaler, train_ds.target_scaler))

        # Average ST predictions across seeds
        st_pred_avg = np.mean(fold_st_preds, axis=0)
        mae_st = np.abs(st_pred_avg - y_val_orig).mean(axis=0)
        print(f"    ST     MAE: visc={mae_st[0]:.2f}, oxid={mae_st[1]:.2f}")
        cv_scores_st.append(mae_st)

        # Hybrid
        hybrid_pred = ST_WEIGHT * st_pred_avg + RIDGE_WEIGHT * ridge_pred_orig
        mae_hybrid = np.abs(hybrid_pred - y_val_orig).mean(axis=0)
        print(f"    Hybrid MAE: visc={mae_hybrid[0]:.2f}, oxid={mae_hybrid[1]:.2f}")
        cv_scores_hybrid.append(mae_hybrid)

        # Optimize weights on this fold
        best_w = ST_WEIGHT
        best_mae_total = mae_hybrid.mean()
        for w in np.arange(0.3, 1.01, 0.05):
            h = w * st_pred_avg + (1 - w) * ridge_pred_orig
            m = np.abs(h - y_val_orig).mean()
            if m < best_mae_total:
                best_mae_total = m
                best_w = w
        print(f"    Best ST weight: {best_w:.2f}")

        fold_st_models.append((fold_seed_models, fold_seed_scalers))
        fold_scalers.append(fold_seed_scalers)

    cv_st = np.array(cv_scores_st)
    cv_ridge = np.array(cv_scores_ridge)
    cv_hybrid = np.array(cv_scores_hybrid)
    print(f"\n  ═══ Средние MAE по {N_FOLDS} фолдам ═══")
    print(f"  Ridge:  visc={cv_ridge[:, 0].mean():.2f}±{cv_ridge[:, 0].std():.2f}  "
          f"oxid={cv_ridge[:, 1].mean():.2f}±{cv_ridge[:, 1].std():.2f}")
    print(f"  ST:     visc={cv_st[:, 0].mean():.2f}±{cv_st[:, 0].std():.2f}  "
          f"oxid={cv_st[:, 1].mean():.2f}±{cv_st[:, 1].std():.2f}")
    print(f"  Hybrid: visc={cv_hybrid[:, 0].mean():.2f}±{cv_hybrid[:, 0].std():.2f}  "
          f"oxid={cv_hybrid[:, 1].mean():.2f}±{cv_hybrid[:, 1].std():.2f}")

    print(f"\n[5/6] Предсказания на тесте...")
    all_st_preds = []
    all_ridge_preds = []

    for visc_model, oxid_model, hc_sc, y_sc_visc, y_sc_oxid in fold_ridge_models:
        hc_test_s = hc_sc.transform(hc_test)
        pred_visc = y_sc_visc.inverse_transform(
            visc_model.predict(hc_test_s).reshape(-1, 1)).ravel()
        pred_oxid = y_sc_oxid.inverse_transform(
            oxid_model.predict(hc_test_s).reshape(-1, 1)).ravel()
        pred_orig = np.column_stack([log_inverse(pred_visc), pred_oxid])
        all_ridge_preds.append(pred_orig)

    ridge_ensemble = np.mean(all_ridge_preds, axis=0)

    for fold_models, fold_scaler_list in fold_st_models:
        for model, (feat_sc, tgt_sc) in zip(fold_models, fold_scaler_list):
            test_copy = []
            for s in test_scenarios:
                sc = {
                    "components": feat_sc.transform(s["components"]).astype(np.float32),
                    "comp_ids": s["comp_ids"],
                    "scenario_id": s["scenario_id"],
                }
                test_copy.append(sc)

            test_comps = [torch.tensor(s["components"], dtype=torch.float32) for s in test_copy]
            test_ids = [torch.tensor(s["comp_ids"], dtype=torch.long) for s in test_copy]
            max_len = max(t.shape[0] for t in test_comps)
            feat_d = test_comps[0].shape[1]
            padded = torch.zeros(len(test_comps), max_len, feat_d)
            padded_ids = torch.zeros(len(test_comps), max_len, dtype=torch.long)
            mask_t = torch.zeros(len(test_comps), max_len, dtype=torch.bool)
            for i, (t, tid) in enumerate(zip(test_comps, test_ids)):
                n = t.shape[0]
                padded[i, :n] = t
                padded_ids[i, :n] = tid
                mask_t[i, :n] = True

            with torch.no_grad():
                model.eval()
                pred, _ = model(padded.to(DEVICE), padded_ids.to(DEVICE), mask_t.to(DEVICE))
                pred = tgt_sc.inverse_transform(pred.cpu().numpy())
                pred_orig = pred.copy()
                pred_orig[:, 0] = log_inverse(pred[:, 0])
                all_st_preds.append(pred_orig)

    st_ensemble = np.mean(all_st_preds, axis=0)
    hybrid_ensemble = ST_WEIGHT * st_ensemble + RIDGE_WEIGHT * ridge_ensemble

    print("\n[6/6] Сохранение predictions.csv...")
    test_sids = [s["scenario_id"] for s in test_scenarios]
    pred_df = pd.DataFrame({
        "scenario_id": test_sids,
        TARGET_COLS[0]: hybrid_ensemble[:, 0],
        TARGET_COLS[1]: hybrid_ensemble[:, 1],
    })
    pred_df.to_csv("predictions.csv", index=False)
    print(f"  Сохранено {len(pred_df)} предсказаний")

    print(f"\n  Статистика предсказаний:")
    print(f"    visc: [{hybrid_ensemble[:, 0].min():.1f}, {hybrid_ensemble[:, 0].max():.1f}], "
          f"mean={hybrid_ensemble[:, 0].mean():.1f}")
    print(f"    oxid: [{hybrid_ensemble[:, 1].min():.1f}, {hybrid_ensemble[:, 1].max():.1f}], "
          f"mean={hybrid_ensemble[:, 1].mean():.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()