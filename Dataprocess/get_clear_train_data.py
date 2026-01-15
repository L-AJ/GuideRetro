import json
import os

from typing import Optional,Set, List, Dict
from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')

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


# ====================== 加载测试集 ======================
def load_test_set(path: str) -> Set[str]:
    print(f"正在加载测试集 ← {path}")
    with open(path, "r", encoding="utf-8") as f:
        test_set = {line.strip() for line in f if line.strip()}
    print(f"测试集分子总数：{len(test_set):,}")
    return test_set


# ====================== 检查反应字符串是否含测试集分子 ======================
def reaction_contains_test_mol(reaction_str: str, test_set: Set[str]) -> bool:
    if ">>" not in reaction_str:
        return False
    left, right = reaction_str.split(">>", 1)
    all_smiles = [s.strip() for s in (left + "." + right).split(".") if s.strip()]
    for smi in all_smiles:
        cano = canonical_smiles_strict_inchi(smi)
        if cano and cano in test_set:
            return True
    return False


# ====================== 主函数 ======================
def clean_and_merge_train_jsons(
    train_json_paths: List[str],
    test_smiles_txt: str,
    output_json: str = "Data/Train/clean_merged_train_FINAL.json"
):
    test_set = load_test_set(test_smiles_txt)

    merged_clean: Dict[str, dict] = {}
    stats = {
        "total_raw": 0,
        "removed_target": 0,
        "removed_route": 0,
        "removed_material": 0,
        "kept": 0,
        "key_changed": 0,        # ← 新增：原始 key ≠ canonical key
        "key_unchanged": 0
    }

    for json_path in train_json_paths:
        print(f"\n{'='*30} 正在处理：{json_path} {'='*30}")
        if not os.path.exists(json_path):
            print("文件不存在，跳过")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_target, item in tqdm(data.items(), desc="清洗 & 统计", unit="mol"):
            stats["total_raw"] += 1

            raw_target = raw_target.strip()
            cano_target = canonical_smiles_strict_inchi(raw_target)
            if not cano_target:
                continue

            # 1. target 在测试集 → 删除整条
            if cano_target in test_set:
                stats["removed_target"] += 1
                continue

            # 2. 检查 retro_routes
            retro_routes = item.get("retro_routes", [])
            if not isinstance(retro_routes, list):
                retro_routes = [retro_routes] if retro_routes else []

            route_contaminated = False
            for route in retro_routes:
                if not isinstance(route, list):
                    route = [route]
                for rxn in route:
                    if isinstance(rxn, str) and reaction_contains_test_mol(rxn, test_set):
                        route_contaminated = True
                        break
                if route_contaminated:
                    break
            if route_contaminated:
                stats["removed_route"] += 1
                continue

            # 3. 检查 materials
            materials = item.get("materials", [])
            if not isinstance(materials, list):
                materials = [materials] if materials else []

            material_contaminated = False
            for mat in materials:
                if isinstance(mat, str):
                    cano = canonical_smiles_strict_inchi(mat.strip())
                    if cano and cano in test_set:
                        material_contaminated = True
                        break
            if material_contaminated:
                stats["removed_material"] += 1
                continue

            # 全部干净 → 保留
            merged_clean[cano_target] = item
            stats["kept"] += 1

            # 统计 key 是否发生变化
            if cano_target != raw_target:
                stats["key_changed"] += 1
            else:
                stats["key_unchanged"] += 1

    # ====================== 保存 ======================
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_clean, f, indent=2, ensure_ascii=False)

    # ====================== 最终报告 ======================
    print("\n" + "="*88)
    print("                     终极零污染清洗 + 合并 + 统计报告")
    print("="*88)
    print(f"原始总样本数量             ：{stats['total_raw']:,}")
    print(f"因 target 在测试集删除      ：{stats['removed_target']:,}")
    print(f"因 retro_routes 污染删除    ：{stats['removed_route']:,}")
    print(f"因 materials 污染删除       ：{stats['removed_material']:,}")
    print(f"─"*50)
    print(f"最终保留干净样本            ：{stats['kept']:,}")
    print(f"其中 key 发生标准化变化的    ：{stats['key_changed']:,}  ({stats['key_changed']/max(stats['kept'],1)*100:5.2f}%)")
    print(f"其中 key 完全不变的         ：{stats['key_unchanged']:,}  ({stats['key_unchanged']/max(stats['kept'],1)*100:5.2f}%)")
    print(f"干净训练集保存路径          → {os.path.abspath(output_json)}")
    print("="*88)
    print("零污染验证命令（必须输出 0）：")
    print(f"grep -f {test_smiles_txt} {output_json} | wc -l")
    print("="*88)


# ====================== 一键运行 ======================
if __name__ == "__main__":
    TRAIN_JSONS = [
        "Data/Train/for_model/train_canolize_dataset.json"
    
    ]

    TEST_SMILES = "final_test_smiles.txt"
    OUTPUT_JSON = "Data/Train/for_model/clean_train_FINAL.json"

    clean_and_merge_train_jsons(
        train_json_paths=TRAIN_JSONS,
        test_smiles_txt=TEST_SMILES,
        output_json=OUTPUT_JSON
    )