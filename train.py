import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

from settings import (
    PATH_RAW_TRAIN, PATH_RAW_PROPS, DEVICE, BATCH_SIZE, EPOCHS, LR, 
    WEIGHT_DECAY, PATIENCE, N_ENSEMBLE_SEEDS, SEED,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, N_SEEDS, DROPOUT
)
from src.data.processing import build_scenarios, DataPreprocessor
from src.data.dataset import DOTDataset, collate_fn
from src.models.set_transformer import SetTransformerDOT
from src.utils.metrics import custom_mae_loss

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights/from_train")

if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Сохраняем веса обучения в папку: {WEIGHTS_DIR}")

    print("="*75)
    print("1. Подготовка данных...")
    print("="*75)

    print("Используем препроцессор для обработки сырых файлов и аугментации")
    preprocessor = DataPreprocessor(props_path=PATH_RAW_PROPS, mode='train')
    train_aug_df, val_clean_df, feature_cols, comp_to_idx = preprocessor.build_train_dataset(PATH_RAW_TRAIN)

    train_scenarios = build_scenarios(train_aug_df, comp_to_idx, feature_cols, is_train=True)
    val_scenarios = build_scenarios(val_clean_df, comp_to_idx, feature_cols, is_train=True)

    for seed_i in range(N_ENSEMBLE_SEEDS):
        s = SEED + seed_i * 137
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

        print(" --- Инициализация датасетов и DataLoader --- ")
        train_ds = DOTDataset(train_scenarios, fit_scalers=True, augment=True)
        val_ds = DOTDataset(val_scenarios, train_ds.feat_scaler, train_ds.global_scaler, train_ds.target_scaler, augment=False)
        
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        print(" --- Инициализация модели --- ")
        model = SetTransformerDOT(
            len(feature_cols) + 1, 
            len(comp_to_idx), 
            D_MODEL, N_HEADS, N_LAYERS, D_FF, N_SEEDS, DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.1
        )

        best_loss, best_state, patience_counter, best_epoch = float("inf"), None, 0, 0
        history = {'train': [], 'val': []}
        
        print(f"\n{'='*75}\nTRAINING SEED {seed_i+1} | PATIENCE: {PATIENCE}\n{'='*75}")
        print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Val Loss':^12} | {'Best Loss':^12} | {'Patience'}")
        print("-" * 75)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            tr_loss = 0
            for b in train_loader:
                optimizer.zero_grad()
                p, _ = model(b[0].to(DEVICE), b[1].to(DEVICE), b[2].to(DEVICE), b[3].to(DEVICE))
                loss = custom_mae_loss(p, b[4].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item() * b[0].shape[0]
            
            tr_loss /= len(train_ds)

            # Валидация
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for b in val_loader:
                    p, _ = model(b[0].to(DEVICE), b[1].to(DEVICE), b[2].to(DEVICE), b[3].to(DEVICE))
                    val_loss += custom_mae_loss(p, b[4].to(DEVICE)).item() * b[0].shape[0]
            val_loss /= len(val_ds)
            
            history['train'].append(tr_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss, patience_counter, best_epoch = val_loss, 0, epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else: 
                patience_counter += 1

            # Логирование обучения и валидации
            do_print = False
            if epoch <= 5: 
                do_print = True
            elif epoch <= 25 and epoch % 5 == 0: 
                do_print = True
            elif epoch > 25 and epoch % 25 == 0: 
                do_print = True
            
            # Печатаем прогресс обучения
            if do_print or patience_counter >= PATIENCE:
                print(f"{epoch:^8} | {tr_loss:^12.4f} | {val_loss:^12.4f} | {best_loss:^12.4f} | {patience_counter:^3}/{PATIENCE}")

            # Early Stopping
            if patience_counter >= PATIENCE:
                print(f"\n[STP] Раннее завершение на эпохе {epoch}. Лучшая эпоха: {best_epoch}")
                break

        print(" --- СОХРАНЕНИЕ АРТЕФАКТОВ ---")
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), f"{WEIGHTS_DIR}/model_seed_{seed_i}.pth")
        joblib.dump(train_ds.feat_scaler, f"{WEIGHTS_DIR}/feat_scaler_seed_{seed_i}.pkl")
        joblib.dump(train_ds.global_scaler, f"{WEIGHTS_DIR}/global_scaler_seed_{seed_i}.pkl")
        joblib.dump(train_ds.target_scaler, f"{WEIGHTS_DIR}/target_scaler_seed_{seed_i}.pkl")

        joblib.dump({
        'len_feature_cols': len(feature_cols),
        'len_comp_to_idx': len(comp_to_idx),
        'feature_cols': feature_cols,
        'comp_to_idx': comp_to_idx
    }, f"{WEIGHTS_DIR}/model_config.pkl")

    print(f"Обучение завершено. Выход из train.py")