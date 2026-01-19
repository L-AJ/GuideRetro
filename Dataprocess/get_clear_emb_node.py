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

def canonical_smiles_strict_inchi(smiles: str) -> Optional[str]:
    if not smiles or not smiles.strip():
        return None

    orig_smiles = smiles.strip()  

    try:
        # Step 1: SMILES → Mol
        mol = Chem.MolFromSmiles(orig_smiles)
        if mol is None:
            return orig_smiles  

        # Step 2: Mol → InChI
        inchi_str = Chem.MolToInchi(mol)
        if not inchi_str:
            return orig_smiles

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


@lru_cache(maxsize=1_000_000)  
def _cached_cano(smiles: str) -> Optional[str]:
    return canonical_smiles_strict_inchi(smiles)

def parallel_canonicalize(smiles_list: List[str]) -> List[Optional[str]]:
    
    num_workers = max(1, cpu_count() - 1)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_cached_cano, smiles_list),
            total=len(smiles_list),
            desc=f" ({num_workers}cores)",
            unit="mol",
            smoothing=0.1
        ))
    return results



def generate_clean_test_set(json_path: str, pkl_paths: List[str], output_txt: str = "final_test_smiles.txt"):
    print("\n" + "="*80)
    


    raw_targets = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_targets.extend(str(k).strip() for k in data.keys() if str(k).strip())

    for pkl in pkl_paths:
        print(f"load PKL ← {pkl}")
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

    
    cano_results = parallel_canonicalize(raw_targets)
    print(len(cano_results))
    test_set: Set[str] = {c for c in cano_results if c}

    with open(output_txt, "w", encoding="utf-8") as f:
        for smi in sorted(test_set):
            f.write(smi + "\n")

    print(f"save → {os.path.abspath(output_txt)} | nums: {len(test_set):,}")
    return output_txt, test_set


def clean_csv_reactions(csv_paths: List[str], test_set: Set[str]):
    csv_dir = os.path.dirname(csv_paths[0]) if csv_paths else "."
    os.makedirs(csv_dir, exist_ok=True)

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

    print(f"reactions: {len(reactions):,}")
    print(f"all_raw_smiles: {len(all_raw_smiles):,}")

    raw_list = list(all_raw_smiles)
   
    cano_list = parallel_canonicalize(raw_list)

    all_canonical_smiles = {}
    for raw, cano in zip(raw_list, cano_list):
        if cano:
            all_canonical_smiles[raw] = cano

    

    forbidden_smiles = test_set

   
    clean_smiles = []
    for cano in all_canonical_smiles.values():
        if cano not in forbidden_smiles:
            clean_smiles.append(cano)

    
    clean_smiles = sorted(set(clean_smiles))

    
    clean_mol_dict = {smi: idx for idx, smi in enumerate(clean_smiles)}

    total_clean = len(clean_mol_dict)
    removed_mols = len(all_canonical_smiles) - total_clean - len(all_canonical_smiles.values()) + len(set(all_canonical_smiles.values()))  
    removed_mols = len(set(all_canonical_smiles.values())) - total_clean  

    print(f"total_clean: {total_clean:,}")
    print(f"removed_mols: {removed_mols:,} 个")
    print(f"ID  0 ~ {total_clean-1}")

    
    clean_lines = []
    removed_rxn = 0

    for reactants, products in tqdm(reactions, desc="Filtering & Splitting", unit="rxn"):
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
            else:  
                line = "\t".join([str(clean_mol_dict[r_cano]), "0", *map(str, [clean_mol_dict[p] for p in p_canos])])
                clean_lines.append(line)

    print(f"delete：{removed_rxn:,}")
    print(f"finally：{len(clean_lines):,}")

   
    mol_file = os.path.join(csv_dir, "all_molecules_clean.txt")
    rxn_file = os.path.join(csv_dir, "clean_reactions.txt")

    with open(mol_file, "w", encoding="utf-8") as f:
        for smi, mid in sorted(clean_mol_dict.items(), key=lambda x: x[1]):
            f.write(f"{mid}\t{smi}\n")

    with open(rxn_file, "w", encoding="utf-8") as f:
        for line in clean_lines:
            f.write(line + "\n")

    print(f"done！")
    print(f"   mols → {mol_file}")
    print(f"   reactions → {rxn_file}")
    print("="*80)

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

    
    test_file, test_set = generate_clean_test_set(TEST_JSON, TEST_PKLS, "final_test_smiles.txt")
    clean_csv_reactions(CSV_FILES, test_set)

    
