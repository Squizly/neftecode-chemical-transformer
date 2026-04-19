import pandas as pd
import numpy as np
import joblib
import torch
import os
from torch.utils.data import DataLoader

from settings import *
from src.data.processing import build_component_vocab, get_feature_columns, build_scenarios, DataPreprocessor
from src.data.dataset import DOTDataset, collate_fn
from src.models.set_transformer import SetTransformerDOT

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights/wd-40")

if __name__ == "__main__":

    print(f"Инференс использует веса из папки: {WEIGHTS_DIR}")

    print("--- Запуск инференса ---")

    print(f"Загрузка конфигурации модели...")
    config = joblib.load(f"{WEIGHTS_DIR}/model_config.pkl")

    feature_cols = config['feature_cols']
    comp_to_idx = config['comp_to_idx']

    len_feature_cols = config['len_feature_cols']
    len_comp_to_idx = config['len_comp_to_idx']

    rename_map={
        'COMP_Ca_cnt': 'cnt_Ca', 
        'COMP_S_cnt': 'cnt_S', 
        'COMP_Zn_cnt': 'cnt_Zn', 
        'CHEM_logp': 'logp', 
        'CHEM_mol_wt': 'mol_wt', 
        'CHEM_rings': 'rings'
    }

    print(f"Препроцессинг данных...")
    preprocessor = DataPreprocessor(props_path=PATH_RAW_PROPS, mode="inference")

    test_df = preprocessor.build_test_dataset(PATH_RAW_TEST)    
    test_df.rename(columns=rename_map, inplace=True)

    print(f"Сборка сценариев...")
    test_scenarios = build_scenarios(test_df, comp_to_idx, feature_cols, is_train=False)
    print(f"Всего сценариев для обработки: {len(test_scenarios)}")

    all_preds = []

    print(f"Начало цикла ансамблирования (N_SEEDS={N_ENSEMBLE_SEEDS})...")
    for seed_i in range(N_ENSEMBLE_SEEDS):
        print(f"Обработка модели {seed_i + 1}/{N_ENSEMBLE_SEEDS}")
        f_sc = joblib.load(f"{WEIGHTS_DIR}/feat_scaler_seed_{seed_i}.pkl")
        g_sc = joblib.load(f"{WEIGHTS_DIR}/global_scaler_seed_{seed_i}.pkl")
        t_sc = joblib.load(f"{WEIGHTS_DIR}/target_scaler_seed_{seed_i}.pkl")

        model = SetTransformerDOT(
            config['len_feature_cols']+1,
            config['len_comp_to_idx'],
            D_MODEL,
            N_HEADS,
            N_LAYERS,
            D_FF,
            N_SEEDS,
            DROPOUT
        ).to(DEVICE)

        model.load_state_dict(torch.load(f"{WEIGHTS_DIR}/model_seed_{seed_i}.pth", map_location=DEVICE))
        model.eval()

        test_ds = DOTDataset(test_scenarios, feat_scaler=f_sc, global_scaler=g_sc, target_scaler=t_sc, augment=False)
        loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        preds = []

        num_batches = len(loader)
        with torch.no_grad():
            for i, b in enumerate(loader):
                p, _ = model(b[0].to(DEVICE), b[1].to(DEVICE), b[2].to(DEVICE), b[3].to(DEVICE))
                preds.append(t_sc.inverse_transform(p.cpu().numpy()))
                if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                    print(f"Сид {seed_i}: Обработано батчей {i+1}/{num_batches}")
        
        all_preds.append(np.concatenate(preds))

    final_preds = np.maximum(np.mean(all_preds, axis=0), 0.0)

    pred_df = pd.DataFrame({
        "scenario_id": [s["scenario_id"] for s in test_scenarios],
        TARGET_COLS_SUBMISSION[0]: final_preds[:, 0],
        TARGET_COLS_SUBMISSION[1]: final_preds[:, 1]
    })

    print(f" Сохранение результатов...")
    os.makedirs("results", exist_ok=True)

    test_df.to_csv("results/test_processed.csv", index=False)

    print("Обработанный файл test.csv сохранен как results/test_processed.csv")

    pred_df.to_csv("results/predictions.csv", index=False)
    print("Файл предсказаний сохранен как results/predictions.csv")