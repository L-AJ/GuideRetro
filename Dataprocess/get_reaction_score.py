import os
import sys
from tqdm import tqdm
from standalone_model_numpy import SCScorer
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from functools import lru_cache

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')



_apply_network = None

@lru_cache(maxsize=200000)
def ultra_fast_scscore(smi: str) -> float:
    if not smi or not smi.strip():
        return 0.0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
    arr = np.zeros((2048,), dtype=np.float32)
    for i in fp.GetOnBits():
        arr[i] = 1.0
    return float(_apply_network(arr))


def main(molecules_file: str, reactions_file: str):
    global _apply_network

    if not os.path.exists(molecules_file):
        print(f"Error:  → {molecules_file}")
        sys.exit(1)
    if not os.path.exists(reactions_file):
        print(f"Error:  → {reactions_file}")
        sys.exit(1)

    out_file = os.path.splitext(reactions_file)[0] + "_scscore.txt"

    raw_lines = []
    with open(molecules_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            raw_lines.append((parts[0], parts[1]))

    total_mols = len(raw_lines)
    print(f"load {total_mols} total_mols.\n")

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.ckpt-10654.as_numpy.json.gz")
    if not os.path.exists(model_path):
        print(f"Error: not find {model_path}")
        sys.exit(1)

    
    model = SCScorer()
    model.restore(model_path, FP_len=2048)
    _apply_network = model.apply  
    print("scscore loaded\n")

    entity_score = {}
    fail_count = 0

    for idx_str, smiles in tqdm(raw_lines, desc="SCScore", unit="mol", ncols=100):
        try:
            idx = int(idx_str)
        except:
            fail_count += 1
            continue
        score = ultra_fast_scscore(smiles)
        entity_score[idx] = score
        if score == 0.0:
            fail_count += 1


    processed = 0
    positive_count = 0  

    with open(reactions_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Reactions", unit="rxn", ncols=100):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            except:
                continue

            diff = entity_score.get(h, 0.0) - entity_score.get(t, 0.0)
            fout.write("\t".join([str(h), str(r), str(t), f"{diff:.4f}"]) + "\n")
            processed += 1
            if diff > 0:
                positive_count += 1

    positive_ratio = positive_count / processed * 100 if processed > 0 else 0



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("use: python get_reaction_score.py <molecules.txt> <reactions.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

    # python Dataprocess/get_reaction_score.py Data/Train/for_embedding/all_molecules.txt Data/Train/for_embedding/clean_reactions.txt
