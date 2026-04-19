import warnings
import joblib
import os
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors
from rdkit import Chem

from settings import DOSE_COL, TEMP_COL, TIME_COL, BIO_COL, CAT_COL, TARGET_COLS_INTERNAL, SEED

warnings.filterwarnings('ignore')

RENAME_MAP = {
    'scenario_id': 'scenario_id', 'Компонент': 'component', 'Наименование партии': 'batch',
    'Массовая доля, %': 'mass_share', 'Количество биотоплива | - Daimler Oxidation Test (DOT), % масс': 'biofuel',
    'Дозировка катализатора, категория': 'catalyst', 'Время испытания | - Daimler Oxidation Test (DOT), ч': 'time',
    'Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C': 'temp',
    'Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %': 'target_visc',
    'Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm': 'target_oxid'
}

MANUAL_SMILES_DB = {
    "125643-61-0": "CCC(CC)COC(=O)CCC1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C", 
    "68411-46-1": "CC(C)(C)CC(C)(C)C1=CC=C(C=C1)NC2=CC=C(C=C2)C(C)(C)CC(C)(C)C", 
    "84605-20-9": "CCCC(CC)COP(=S)([O-])OCC(CC)CCCC.[Zn+2]", 
    "134758-95-5": "CCCCCCCCCCCC1CC(=O)N(C1=O)CCN", 
}

CLASS_TEMPLATES = {
    'Антиоксидант': 'CC(C)(C)C1=CC(=CC(=C1O)C(C)(C)C)CCC(=O)OC',
    'Антипенная': 'C[Si](C)(C)O[Si](C)(C)O[Si](C)(C)C',
    'Депрессорная': 'CC(=C)C(=O)OCCCCCCCCCCCC',
    'Дисперсант': 'CCCCCCCCCCCC1CC(=O)N(C1=O)CCN',
    'Загуститель': 'CCCCCCCCCCCCCCCCCCCC',
    'Противоизносная': 'CC(C)COP(=S)([O-])OCC(C)C.[Zn+2]',
    'Детергент': 'CCCCCCCCCCCC1=CC=C(S(=O)(=O)O[Ca])C=C1',
    'Соединение_молибдена': 'CCCCCCCCOP(=S)(S)S[Mo](=S)S'
}

# Строго заданный порядок колонок для Test
TEST_GOLDEN_COLUMNS = [
    'scenario_id', 'component', 'temp', 'time', 'biofuel', 'catalyst',
    'mass_norm', 'hidden_pct', 'logp', 'mol_wt', 'rings', 'cnt_Ca', 'cnt_S',
    'cnt_Zn', 'COMP_total_metals',
    'COMP_Активный Азот / Кислород, % масс. (N или O)_NoMethod',
    'COMP_Кислотное число_ГОСТ 11362',
    'COMP_Массовая доля кальция_ASTM D6481',
    'COMP_Массовая доля серы_ASTM D6481',
    'COMP_Массовая доля фосфора_ASTM D6481',
    'COMP_Массовая доля цинка_ASTM D6481',
    'COMP_Общее содержание азота_ASTM D3228',
    'COMP_Содержание MgCO3, CaCO3, % масс._NoMethod',
    'COMP_Содержание Азота_NoMethod', 'COMP_Содержание Бора_NoMethod',
    'COMP_Содержание воды, % масс._NoMethod',
    'COMP_Содержание масла, % масс._NoMethod',
    'COMP_Содержание масла_NoMethod',
    'COMP_Содержание металла (Ca/Mg), % масс._NoMethod',
    'COMP_Содержание мыла, % масс._NoMethod',
    'COMP_Содержание насыщ. у/в_NoMethod',
    'COMP_Содержание серы, % масс._NoMethod',
    'COMP_Содержание серы, мг/кг_NoMethod',
    'COMP_Щелочное число_ASTM D2896', 'COMP_Щелочное число_ГОСТ 11362',
    'DENS_Плотность при 15°С_ASTM D4052',
    'DENS_Плотность при 20°С_ASTM D4052',
    'ENER_Потенциал ионизации,эВ_NoMethod',
    'ENER_Химический потенциал, Дж/моль_NoMethod',
    'ENER_Энергия ВЗМО, эВ_NoMethod', 'ENER_Энергия НСМО, эВ_NoMethod',
    'ENER_Энергия диссоциации связи Х-Н, ккал/моль_NoMethod',
    'OTHER_bio_risk', 'OTHER_severity_idx',
    'OTHER_Атомное отношение P:Zn_NoMethod', 'OTHER_Группа по API_NoMethod',
    'OTHER_Деэм.вода_ASTM D1401', 'OTHER_Деэм.время_ASTM D1401',
    'OTHER_Деэм.масло_ASTM D1401', 'OTHER_Деэм.эмульсия_ASTM D1401',
    'OTHER_Дипольный момент, Д_NoMethod',
    'OTHER_Длина углеродной цепи_NoMethod',
    'OTHER_Испаряемость по NOACK_ASTM D5800',
    'OTHER_Масса гидрофобного хвоста, г/моль_NoMethod',
    'OTHER_Номер CAS / SMILES_NoMethod', 'OTHER_Номер CAS_NoMethod',
    'OTHER_Определение содержания воды методом Карла Фишера  ASTM D 6304_NoMethod',
    'OTHER_Отношение Мыло/Основание_NoMethod',
    'OTHER_Последовательность 1_ASTM D892',
    'OTHER_Последовательность 2_ASTM D892',
    'OTHER_Последовательность 3_ASTM D892',
    'OTHER_Разветвленность радикала / радикалов_NoMethod',
    'OTHER_Размер мицелл, нм_NoMethod',
    'OTHER_Степень полисульфидности_NoMethod',
    'OTHER_Стерический фактор, Å3_NoMethod',
    'OTHER_Структура УВ-радикала_NoMethod', 'OTHER_Цвет_ASTM D1500',
    'TEMP_Температура застывания_ГОСТ 20287',
    'TEMP_Температура плавления, °C_NoMethod', 'VISC_index_ratio',
    'VISC_Динамическая вязкость CCS -15°C_ASTM D5293',
    'VISC_Динамическая вязкость CCS -20°C_ASTM D5293',
    'VISC_Динамическая вязкость CCS -25°C_ASTM D5293',
    'VISC_Динамическая вязкость CCS -30°C_ASTM D5293',
    'VISC_Динамическая вязкость CCS -35°C_ASTM D5293',
    'VISC_Индекс вязкости_ГОСТ 25371',
    'VISC_Индекс стабильности, %_NoMethod',
    'VISC_Кинематическая вязкость, при 100°C_ASTM D445',
    'VISC_Кинематическая вязкость, при 40°C_ASTM D445',
    'VISC_Кинематическая вязкость_NoMethod'
]

# Для Train добавляем в конец целевые переменные
GOLDEN_COLUMNS = TEST_GOLDEN_COLUMNS + ['target_visc', 'target_oxid', 'target_visc_log']

def build_component_vocab(train_df, test_df):
    all_comps = sorted(set(train_df["component"].unique()) | set(test_df["component"].unique()))
    return {c: i + 1 for i, c in enumerate(all_comps)}

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
            raw_doses.append(row[DOSE_COL])
            comp_ids.append(comp_to_idx.get(comp_name, 0))
            feats = row[feature_cols].values.astype(np.float32)
            comp_features.append(np.concatenate([[row[DOSE_COL]], feats]))
        first_row = grp.iloc[0]
        global_feats = np.array([first_row['temp'], first_row['time'], first_row['biofuel'], first_row['catalyst']], dtype=np.float32)
        scenario = {"components": np.stack(comp_features), "comp_ids": np.array(comp_ids, dtype=np.int64),
                    "global_feats": global_feats, "raw_doses": np.array(raw_doses, dtype=np.float32), "scenario_id": sid}
        if is_train:
            scenario["targets"] = np.array([first_row[TARGET_COLS_INTERNAL[0]], first_row[TARGET_COLS_INTERNAL[1]]], dtype=np.float32)
        scenarios.append(scenario)
    return scenarios

def parse_numeric(val):
    s = str(val).replace(',', '.').replace('<', '').replace('>', '').strip()
    nums = re.findall(r'-?\d+\.?\d*', s)
    if len(nums) == 2 and '-' in s: return (float(nums[0]) + float(nums[1])) / 2
    return float(nums[0]) if nums else np.nan

def get_phys_category(name):
    n = name.lower()
    if any(x in n for x in ['плотность', 'density']): return 'DENS'
    if any(x in n for x in ['вязкость', 'viscosity', 'индекс']): return 'VISC'
    if any(x in n for x in ['доля', 'содержание', 'состав', 'металл', 'азот', 'сера', 'фосфор', 'кальций', 'цинк', 'щелочное', 'кислотное']): return 'COMP'
    if any(x in n for x in ['температура', 'точка', '°c']): return 'TEMP'
    if any(x in n for x in ['энергия', 'эв', 'потенциал']): return 'ENER'
    return 'OTHER'

def is_valid_smiles(v):
    s = str(v)
    return len(s) > 15 and 'C' in s and ('(' in s or '=' in s) and not re.search(r'[А-Яа-я]', s)

def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return pd.Series([0]*6, index=['mol_wt', 'logp', 'cnt_S', 'cnt_Zn', 'cnt_Ca', 'rings'])
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    return pd.Series({
        'mol_wt': Descriptors.MolWt(mol), 
        'logp': Descriptors.MolLogP(mol),
        'cnt_S': syms.count('S'), 
        'cnt_Zn': syms.count('Zn'), 
        'cnt_Ca': syms.count('Ca'),
        'rings': Descriptors.RingCount(mol)
    })

class DataPreprocessor:
    def __init__(self, props_path, mode='train'):
        self.mode = mode
        
        self.props_raw = pd.read_csv(props_path)
        self.props_raw.columns = ['component', 'batch', 'param', 'unit', 'value']

    def process_properties(self, all_comps):
        props = self.props_raw.copy()
        props['param'] = props['param'].str.replace(' |_', '_', regex=False).str.replace('|', '', regex=False).str.strip()
        props['v'] = props['value'].apply(parse_numeric)
        
        props.loc[props['unit'].str.contains('мг/кг', na=False), 'v'] /= 10000
        props.loc[props['param'].str.contains('Плотность', na=False) & (props['v'] < 2.0), 'v'] *= 1000
        
        props['cat'] = props['param'].apply(lambda x: re.sub(r',? (ASTM D\d+|DIN \d+|ГОСТ \d+).*', '', str(x)).strip()).apply(get_phys_category)
        props['clean_p'] = props['param'].apply(lambda x: re.sub(r',? (ASTM D\d+|DIN \d+|ГОСТ \d+).*', '', str(x)).strip())
        props['final_f_name'] = props['cat'] + "_" + props['clean_p'] + "_" + props.apply(lambda x: re.search(r'(ASTM D\d+|DIN \d+|ГОСТ \d+)', str(x['param'])).group(1) if re.search(r'(ASTM D\d+|DIN \d+|ГОСТ \d+)', str(x['param'])) else "NoMethod", axis=1)
        
        piv_t = props[props['batch'] == 'typical'].pivot_table(index='component', columns='final_f_name', values='v', aggfunc='first')
        piv_a = props[props['batch'] != 'typical'].pivot_table(index=['component', 'batch'], columns='final_f_name', values='v', aggfunc='first').reset_index()
        
        for col in piv_t.columns:
            typical_values = piv_a['component'].map(piv_t[col])
            if col in piv_a.columns: piv_a[col] = piv_a[col].fillna(typical_values)
            else: piv_a[col] = typical_values
            
        comp_features_raw = piv_a.drop(columns=[c for c in piv_a.columns if piv_a[c].nunique() <= 1 and c not in ['component', 'batch']])
        
        s_dict = props[props['value'].apply(is_valid_smiles)].groupby('component')['value'].first().to_dict()
        cas_map = props[props['param'].str.contains('CAS', na=False)].groupby('component')['value'].first().apply(lambda x: re.search(r'\d{2,7}-\d{2}-\d', str(x)).group(0) if re.search(r'\d{2,7}-\d{2}-\d', str(x)) else None).dropna().to_dict()
        
        chem_rows = []
        for c in all_comps:
            s = s_dict.get(c)
            cas = cas_map.get(c)
            if not s and cas in MANUAL_SMILES_DB: s = MANUAL_SMILES_DB[cas]
            if not s: s = CLASS_TEMPLATES.get(str(c).split('_')[0], "CCCCCCCCCCCCCCCCCCCC")
            
            desc = get_rdkit_descriptors(s)
            chem_rows.append({'component': c, **desc.to_dict()})
            
        chem_features = pd.DataFrame(chem_rows)
        self.all_comp_info = comp_features_raw.merge(chem_features, on='component', how='left')

    def finalize_df(self, df_raw, is_train=True):
        df = df_raw.copy()
        df['mass_norm'] = df['mass_share'] / (df.groupby('scenario_id')['mass_share'].transform('sum') + 1e-9)
        df['hidden_pct'] = df['batch'].str.extract(r'(\d+[.,]\d+|\d+)%')[0].str.replace(',','.').astype(float).fillna(0)
        
        res = df.merge(self.all_comp_info, on=['component', 'batch'], how='left')
        
        v40, v100 = 'VISC_Кинематическая вязкость, при 40°C_ASTM D445', 'VISC_Кинематическая вязкость, при 100°C_ASTM D445'
        if v40 in res.columns and v100 in res.columns:
            res['VISC_index_ratio'] = (res[v40] / (res[v100] + 1e-6)).replace([np.inf, -np.inf], 0)
        
        res['OTHER_severity_idx'] = (res['temp'] * res['time']) / 1000
        res['COMP_total_metals'] = res.get('COMP_Массовая доля цинка_ASTM D6481', 0) + res.get('COMP_Массовая доля кальция_ASTM D6481', 0)
        res['OTHER_bio_risk'] = (res['temp'] == 150).astype(int) * res['biofuel']

        targets = []
        if is_train:
            res['target_visc_log'] = np.log1p(res['target_visc'] + 40)
            targets = ['target_visc', 'target_oxid', 'target_visc_log']

        id_cols = ['scenario_id', 'component']

        numeric_df = res.select_dtypes(include=[np.number])
        res_clean = pd.concat([res[id_cols], numeric_df], axis=1)
        res_clean = res_clean.loc[:, ~res_clean.columns.duplicated()]

        return res_clean, targets
    
    def augment_data(self, df, target_scenarios=1000, val_size=50, noise_std=0.02):
        unique_ids = df['scenario_id'].unique()
        train_ids, val_ids = train_test_split(unique_ids, test_size=val_size, random_state=42)
        
        val_df = df[df['scenario_id'].isin(val_ids)].copy()
        train_base = df[df['scenario_id'].isin(train_ids)].copy()
        
        all_train_dfs = [train_base]
        num_base = len(train_ids)
        needed = target_scenarios - num_base
        
        multiplier = (needed // num_base) + 1

        for i in range(1, multiplier + 1):
            aug_copy = train_base.copy()

            noise = np.random.normal(0, noise_std, size=len(aug_copy))
            aug_copy['mass_norm'] += noise
            aug_copy['mass_norm'] = aug_copy['mass_norm'].clip(lower=0.0001)
            
            aug_copy['mass_norm'] = aug_copy['mass_norm'] / aug_copy.groupby('scenario_id')['mass_norm'].transform('sum')
            
            aug_copy['scenario_id'] = aug_copy['scenario_id'].astype(str) + f"_aug_{i}"
            all_train_dfs.append(aug_copy)
            
            if pd.concat(all_train_dfs)['scenario_id'].nunique() >= target_scenarios:
                break

        final_df = pd.concat(all_train_dfs, ignore_index=True)
        final_unique_ids = final_df['scenario_id'].unique()[:target_scenarios]
        final_df = final_df[final_df['scenario_id'].isin(final_unique_ids)]
        
        return final_df, val_df

    def align_schema(self, df, target_columns):
        """Выравнивает датафрейм по эталонному списку колонок."""
        df_aligned = df.copy()
        
        # Удаляем лишние колонки
        extra_cols = set(df_aligned.columns) - set(target_columns)
        if extra_cols:
            df_aligned = df_aligned.drop(columns=list(extra_cols))
            
        # Восстанавливаем недостающие колонки нулями
        missing_cols = set(target_columns) - set(df_aligned.columns)
        for col in missing_cols:
            df_aligned[col] = 0
            
        # Возвращаем В СТРОГО ЗАДАННОМ ПОРЯДКЕ
        return df_aligned[target_columns]

    def build_train_dataset(self, train_path):
        
        train_raw = pd.read_csv(train_path).rename(columns=RENAME_MAP)
        all_comps = train_raw['component'].unique()
        
        comp_to_idx = {c: i + 1 for i, c in enumerate(all_comps)}
        
        self.process_properties(all_comps)
        
        f_train, target_names = self.finalize_df(train_raw, is_train=True)
        
        feature_cols = [c for c in GOLDEN_COLUMNS if c not in ['scenario_id', 'component'] + target_names]
        
        train_aug_df, val_clean_50_df = self.augment_data(f_train, target_scenarios=1000)
        
        train_aligned = self.align_schema(train_aug_df, GOLDEN_COLUMNS).fillna(0)
        val_aligned = self.align_schema(val_clean_50_df, GOLDEN_COLUMNS).fillna(0)
        
        return train_aligned, val_aligned, feature_cols, comp_to_idx

    def build_test_dataset(self, test_path):
        """ВЫЗЫВАЕТСЯ ТОЛЬКО ПРИ ИНФЕРЕНСЕ"""
        test_raw = pd.read_csv(test_path).rename(columns=RENAME_MAP)
        all_comps = test_raw['component'].unique()
        
        # Обработка свойств
        self.process_properties(all_comps)
        f_test, _ = self.finalize_df(test_raw, is_train=False)
        
        # Вызываем метод выравнивания строго по TEST_GOLDEN_COLUMNS
        f_test = self.align_schema(f_test, TEST_GOLDEN_COLUMNS).fillna(0)
        
        return f_test