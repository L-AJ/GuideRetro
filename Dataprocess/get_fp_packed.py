import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool, cpu_count
import rdkit.RDLogger as RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')


FP_RADIUS = 2
FP_BITS = 2048
PACKED_SIZE = FP_BITS // 8  # 2048 / 8 = 256 bytes

def process_batch_packed(smiles_list):
    batch_data = []
    np_fp = np.zeros((FP_BITS,), dtype=np.uint8)
    
    for smiles in smiles_list:
        is_success = False
        if not (pd.isna(smiles) or str(smiles).strip() == ''):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
                    
                    # 1. RDKit -> Numpy (0/1 array)
                    DataStructs.ConvertToNumpyArray(fp, np_fp)
                    
                    # 2. Numpy -> Packed Bits 
                    # [0, 1, 0, 1, 0, 0, 0, 0] -> [integer]
                    packed_row = np.packbits(np_fp)
                   
                    batch_data.append(packed_row.tobytes())
                    is_success = True
                except:
                    pass

        if not is_success:
            batch_data.append(None)
            
    return batch_data

def generate_fingerprints_optimized(data_dir, filename='all_molecules_clean.txt', num_workers=None):
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(data_dir, 'fingerprints_packed.npy')
    
    print(f"[*] Reading data from: {input_path}")
    df = pd.read_csv(input_path, sep='\t', header=None, names=['id', 'smiles'], dtype=str)
    all_smiles = df['smiles'].tolist()
    total_count = len(all_smiles)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    chunk_size = 5000 
    chunks = [all_smiles[i:i + chunk_size] for i in range(0, total_count, chunk_size)]
    
    print(f"[*] Starting parallel processing with {num_workers} workers...")
    print(f"[*] Target Shape: ({total_count}, {PACKED_SIZE}) uint8")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_batch_packed, chunks), 
            total=len(chunks),
            unit="chunk"
        ))
    
    print("[*] Aggregating results...")
    
    final_matrix = np.zeros((total_count, PACKED_SIZE), dtype=np.uint8)
    
    current_idx = 0
    success_count = 0
    
    for batch_data in tqdm(results, desc="Stacking"):
        for packed_bytes in batch_data:
            if packed_bytes is not None:
                final_matrix[current_idx] = np.frombuffer(packed_bytes, dtype=np.uint8)
                success_count += 1
            else:
                pass
            current_idx += 1

    fail_count = total_count - success_count

    print(f"[*] Saving to {output_path} ...")
    np.save(output_path, final_matrix)
    
    file_size_mb = os.path.getsize(output_path) / (1024*1024)
    
    print("\n" + "="*40)
    print(f"Optimization Complete!")
    print(f"Total processed: {total_count}")
    print(f"Success:         {success_count} ({success_count/total_count*100:.2f}%)")
    print(f"Failed (Zeroed): {fail_count}")
    print(f"Final Shape:     {final_matrix.shape}")
    print(f"File size:       {file_size_mb:.2f} MB")
    print(f"Space Savings:   ~87.5% vs unpacked")
    print("="*40)

if __name__ == "__main__":
    TARGET_DIR = "Data/Train/for_embedding"
    
    if os.path.exists(TARGET_DIR):
        generate_fingerprints_optimized(TARGET_DIR)
        
    else:
        print(f"Directory {TARGET_DIR} not found.")
