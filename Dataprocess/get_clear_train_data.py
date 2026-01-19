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
        mol = Chem.MolFromSmiles(orig_smiles)
        if mol is None:
            return orig_smiles 

        # Step 2: Mol → InChI
        inchi_str = Chem.MolToInchi(mol)
        if not inchi_str:
            return orig_smiles

        # Step 3: InChI → Mol
        result = Chem.MolFromInchi(inchi_str)
        if isinstance(result, tuple):
            mol2, retcode = result
        else:
            mol2 = result

        if mol2 is None:
            return orig_smiles

        for atom in mol2.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')

        final = Chem.MolToSmiles(mol2)
        return final if final else orig_smiles

    except Exception as e:
        return orig_smiles


def load_test_set(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        test_set = {line.strip() for line in f if line.strip()}
    return test_set


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
        "key_changed": 0,        
        "key_unchanged": 0
    }

    for json_path in train_json_paths:
        print(f"\n{'='*30} process：{json_path} {'='*30}")
        if not os.path.exists(json_path):
            print("no file，pass!")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_target, item in tqdm(data.items(), desc="doing...", unit="mol"):
            stats["total_raw"] += 1

            raw_target = raw_target.strip()
            cano_target = canonical_smiles_strict_inchi(raw_target)
            if not cano_target:
                continue

            if cano_target in test_set:
                stats["removed_target"] += 1
                continue

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

            merged_clean[cano_target] = item
            stats["kept"] += 1

            if cano_target != raw_target:
                stats["key_changed"] += 1
            else:
                stats["key_unchanged"] += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_clean, f, indent=2, ensure_ascii=False)

    print("\n" + "="*88)
    print(f"save path           → {os.path.abspath(output_json)}")
    print("="*88)
    print("="*88)


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
