import json
import pickle
import csv
import os
from typing import List, Optional, Set
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from functools import lru_cache
from multiprocessing import Pool, cpu_count

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')

# ================== 核弹级 InChI 标准化（修复版）==================
def canonical_smiles_strict_inchi(smiles: str) -> Optional[str]:
    if not smiles or not smiles.strip():
        return None

    orig_smiles = smiles.strip()  

    try:
        # Step 1: 尝试 SMILES → Mol
        mol = Chem.MolFromSmiles(orig_smiles)
        if mol is None:
            return orig_smiles  # 连解析都失败 → 只能原样返回

        # Step 2: Mol → InChI
        inchi_str = Chem.MolToInchi(mol)
        if not inchi_str:
            return orig_smiles

        # Step 3: InChI → Mol（关键修复）
        result = Chem.MolFromInchi(inchi_str)
        if isinstance(result, tuple):
            mol2, retcode = result
        else:
            mol2 = result

        if mol2 is None:
            return orig_smiles

        # Step 4: 清除原子映射号（必须！否则去污染失效）
        for atom in mol2.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

        # Step 5: 生成最严格 canonical SMILES
        final = Chem.MolToSmiles(mol2)
        return final if final else orig_smiles

    except Exception as e:
        return orig_smiles



# ================== 全局缓存 + 多进程加速（核心提速 10x+）==================
@lru_cache(maxsize=1_000_000)  # 100万缓存足够覆盖几乎所有重复
def _cached_cano(smiles: str) -> Optional[str]:
    return canonical_smiles_strict_inchi(smiles)

def parallel_canonicalize(smiles_list: List[str]) -> List[Optional[str]]:
    """多进程 + 缓存并行标准化"""
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_cached_cano, smiles_list),
            total=len(smiles_list),
            desc=f"并行标准化 ({num_workers}核)",
            unit="mol",
            smoothing=0.1
        ))
    return results


# ================== 第一步：生成严格标准化的测试集 ==================
def generate_clean_test_set(json_path: str, pkl_paths: List[str], output_txt: str = "final_test_smiles.txt"):
    print("\n" + "="*80)
    print("第一步：生成严格标准化的测试集分子（InChI 核弹级）")
    print("="*80)

    raw_targets = []
    print(f"读取 JSON ← {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_targets.extend(str(k).strip() for k in data.keys() if str(k).strip())

    for pkl in pkl_paths:
        print(f"读取 PKL ← {pkl}")
        with open(pkl, "rb") as f:
            try:
                data = pickle.load(f)
            except:
                import dill
                f.seek(0)
                data = dill.load(f)
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                s = str(item[0]).strip()
                s = s.split(">>")[0] if ">>" in s else s
                if s:
                    raw_targets.append(s)

    print(f"\n开始并行标准化 {len(raw_targets):,} 个测试集分子...")
    cano_results = parallel_canonicalize(raw_targets)
    print(len(cano_results))
    test_set: Set[str] = {c for c in cano_results if c}

    with open(output_txt, "w", encoding="utf-8") as f:
        for smi in sorted(test_set):
            f.write(smi + "\n")

    print(f"测试集保存 → {os.path.abspath(output_txt)} | 数量: {len(test_set):,}")
    return output_txt, test_set


# ================== 第二步：彻底清洗 CSV（零污染 + 超高速）==================
def clean_csv_reactions(csv_paths: List[str], test_set: Set[str]):
    csv_dir = os.path.dirname(csv_paths[0]) if csv_paths else "."
    os.makedirs(csv_dir, exist_ok=True)

    print("\n" + "="*80)
    print("第二步：彻底清洗 CSV（InChI 核弹级 + 多核并行）")
    print(f"输出目录 → {os.path.abspath(csv_dir)}")
    print("="*80)
    print(f"已加载测试集用于去污染：{len(test_set):,} 个分子")

    # 第一阶段：读取所有反应 + 收集原始 SMILES
    print("第一阶段：读取所有 CSV 并收集原始分子...")
    reactions = []
    all_raw_smiles = set()

    for csv_path in tqdm(csv_paths, desc="读取 CSV", unit="file"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2: continue
                rxn = row[1].strip()
                if ">>" not in rxn: continue
                reactants = [x.strip() for x in rxn.split(">>")[0].split(".") if x.strip()]
                products  = [x.strip() for x in rxn.split(">>")[1].split(".") if x.strip()]
                if not reactants or not products: continue

                reactions.append((reactants, products))
                all_raw_smiles.update(reactants)
                all_raw_smiles.update(products)

    print(f"原始反应数：{len(reactions):,}")
    print(f"发现唯一原始分子：{len(all_raw_smiles):,}")

    # 并行标准化（最慢的部分，提速 10x+）
    raw_list = list(all_raw_smiles)
    print(f"开始并行标准化所有分子（缓存+多核）...")
    cano_list = parallel_canonicalize(raw_list)

    all_canonical_smiles = {}
    for raw, cano in zip(raw_list, cano_list):
        if cano:
            all_canonical_smiles[raw] = cano

    print(f"标准化完成 → 有效分子：{len(all_canonical_smiles):,}")

    # 第二阶段：剔除测试集分子 + 重排为 0~N-1 连续 ID（最重要！）
    print("\n第二阶段：剔除测试集污染分子 + 重排连续 ID...")
    forbidden_smiles = test_set

    # 先收集所有不被禁的 canonical SMILES
    clean_smiles = []
    for cano in all_canonical_smiles.values():
        if cano not in forbidden_smiles:
            clean_smiles.append(cano)

    # 去重（极少数情况下可能有重复）
    clean_smiles = sorted(set(clean_smiles))

    # 重排为从 0 开始的连续整数
    clean_mol_dict = {smi: idx for idx, smi in enumerate(clean_smiles)}

    total_clean = len(clean_mol_dict)
    removed_mols = len(all_canonical_smiles) - total_clean - len(all_canonical_smiles.values()) + len(set(all_canonical_smiles.values()))  # 简化计算
    removed_mols = len(set(all_canonical_smiles.values())) - total_clean  # 正确计算

    print(f"去污染后剩余分子：{total_clean:,}")
    print(f"成功剔除测试集分子：{removed_mols:,} 个")
    print(f"ID 重排完成 → 范围 0 ~ {total_clean-1}（连续无空洞，最佳实践！）")

    # 第三阶段：过滤反应 + 生成训练样本（制表符分割）
    print(f"\n第三阶段：过滤反应并拆分（共 {len(reactions):,} 条）...")
    clean_lines = []
    removed_rxn = 0

    for reactants, products in tqdm(reactions, desc="过滤 & 拆分", unit="rxn"):
        has_forbidden = any(all_canonical_smiles.get(s, "") in forbidden_smiles for s in reactants + products)
        if has_forbidden:
            removed_rxn += 1
            continue

        for r_raw in reactants:
            r_cano = all_canonical_smiles.get(r_raw)
            if not r_cano or r_cano not in clean_mol_dict:
                continue
            p_canos = []
            for p_raw in products:
                p_cano = all_canonical_smiles.get(p_raw)
                if not p_cano or p_cano not in clean_mol_dict:
                    break
                p_canos.append(p_cano)
            else:  # 所有产物都合法
                line = "\t".join([str(clean_mol_dict[r_cano]), "0", *map(str, [clean_mol_dict[p] for p in p_canos])])
                clean_lines.append(line)

    print(f"因污染删除反应：{removed_rxn:,}")
    print(f"最终训练样本数：{len(clean_lines):,}")

    # 输出
    mol_file = os.path.join(csv_dir, "all_molecules.txt")
    rxn_file = os.path.join(csv_dir, "clean_reactions.txt")

    with open(mol_file, "w", encoding="utf-8") as f:
        for smi, mid in sorted(clean_mol_dict.items(), key=lambda x: x[1]):
            f.write(f"{mid}\t{smi}\n")

    with open(rxn_file, "w", encoding="utf-8") as f:
        for line in clean_lines:
            f.write(line + "\n")

    print(f"\n零污染清洗完成！")
    print(f"   分子表 → {mol_file}")
    print(f"   反应表 → {rxn_file}")
    print(f"   验证命令：grep -f final_test_smiles.txt {mol_file} | wc -l")
    print(f"   预期结果：0  ← 必须是 0！")
    print("="*80)


# ================== 主程序 ==================
if __name__ == "__main__":
    TEST_JSON = "Data/Test/test_dataset.json"
    TEST_PKLS = [
        "Data/Test/chembl_1000.pkl",
        "Data/Test/gdb17_1000.pkl",
        "Data/Test/routes_possible_test_hard.pkl"
    ]
    CSV_FILES = [
        "Data/Train/for_embedding/raw_test.csv",
        "Data/Train/for_embedding/raw_train.csv",
        "Data/Train/for_embedding/raw_val.csv",
    ]

    print("开始执行终极零污染 + 超高速清洗流程...")
    test_file, test_set = generate_clean_test_set(TEST_JSON, TEST_PKLS, "final_test_smiles.txt")
    clean_csv_reactions(CSV_FILES, test_set)

    print("\n全部完成！")
    print("验证零污染：")
    print("grep -f final_test_smiles.txt Data/Train/for_embedding/all_molecules.txt | wc -l")
    print("结果必须为：0")