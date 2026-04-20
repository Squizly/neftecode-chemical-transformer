import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from settings import AUG_MULTIPLIER, AUG_NOISE_STD, AUG_DOSE_JITTER, AUG_DROP_THRESHOLD, AUG_DROP_PROB

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
                self.target_scaler = StandardScaler().fit(np.stack([s["targets"] for s in scenarios]))
        else:
            self.feat_scaler = feat_scaler
            self.global_scaler = global_scaler
            self.target_scaler = target_scaler
        
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
        
        gf_scaled = self.global_scaler.transform(gf.reshape(1, -1)).flatten().astype(np.float32)

        if self.has_targets: 
            return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled), torch.tensor(s["targets_scaled"])
        
        return torch.tensor(comps), torch.tensor(comp_ids), torch.tensor(gf_scaled)

def collate_fn(batch):
    has_targets = len(batch[0]) == 4

    components = [b[0] for b in batch]
    comp_ids = [b[1] for b in batch]

    global_feats = torch.stack([b[2] for b in batch])
    
    targets = torch.stack([b[3] for b in batch]) if has_targets else None

    max_len = max(c.shape[0] for c in components)
    
    padded = torch.zeros(len(components), max_len, components[0].shape[1])
    
    padded_ids = torch.zeros(len(components), max_len, dtype=torch.long)
    mask = torch.zeros(len(components), max_len, dtype=torch.bool)

    for i, (c, cid) in enumerate(zip(components, comp_ids)):
        n = c.shape[0]
        padded[i, :n] = c
        padded_ids[i, :n] = cid
        mask[i, :n] = True

    if targets is not None:
        return (padded, padded_ids, global_feats, mask, targets)

    return (padded, padded_ids, global_feats, mask)