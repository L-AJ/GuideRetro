import time
import numpy as np
import faiss
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import os
import multiprocessing
from typing import List, Dict, Tuple, Optional, Union

# -----------------------------------------------------------------------------
# for muti Worker 
# -----------------------------------------------------------------------------
def _process_chunk(chunk_data):
    
    indices, smiles_list = chunk_data
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    fps = []
    popcounts = [] 
    valid_indices = []
    valid_smiles = []
    
    for idx, smi in zip(indices, smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            fp_obj = generator.GetFingerprint(mol)
            
        
            binary_text = DataStructs.BitVectToBinaryText(fp_obj)
            fp_arr = np.frombuffer(binary_text, dtype=np.uint8)
            
            pop = fp_obj.GetNumOnBits()
            
            fps.append(fp_arr)
            popcounts.append(pop)
            valid_indices.append(idx)
            valid_smiles.append(smi)
            
        except Exception:
            continue
            
    return fps, popcounts, valid_indices, valid_smiles

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
class SimilaritySearcher:
    def __init__(
        self, 
        entity_file: str, 
        feature_file: str,
        cache_file: Optional[str] = None,
        use_gpu: bool = True,
        num_workers: int = None
    ):
        self.entity_file = entity_file
        self.feature_file = feature_file
        self.cache_file = cache_file
        self.use_gpu = use_gpu
        self.num_workers = num_workers if num_workers else max(1, multiprocessing.cpu_count() - 2)
        
        # 核心数据
        self.index = None
        self.fingerprints = None    # uint8 array (N, 256)
        self.popcounts = None       # int16 array (N,) 
        self.features = None        # float32 array (N, D)
        self.smiles_list = None     # List[str]
        self.original_indices = None 
        
        self._load_data()
        self._build_index()

    def _load_data(self):
        
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading from cache: {self.cache_file}")
            data = np.load(self.cache_file, allow_pickle=True)
            self.fingerprints = data['fingerprints']
            self.popcounts = data['popcounts']
            self.features = data['features']
            self.original_indices = data['indices']
            self.smiles_list = data['smiles'].tolist()
            print(f"Loaded {len(self.fingerprints)} molecules.")
            return

        print("Cache not found. Processing raw files...")
        
       
        raw_entries = []
        with open(self.entity_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    #  ID \t SMILES
                    raw_entries.append((int(parts[0]), parts[1]))
        
        total_items = len(raw_entries)
        print(f"Total raw entries: {total_items}")
        
        
        chunk_size = max(100, total_items // (self.num_workers * 4))
        chunks = []
        for i in range(0, total_items, chunk_size):
            batch = raw_entries[i:i + chunk_size]
            indices = [x[0] for x in batch]
            smiles = [x[1] for x in batch]
            chunks.append((indices, smiles))
            
        
        fps_list = []
        pop_list = []
        indices_list = []
        smiles_list = []
        
        print(f"Processing fingerprints with {self.num_workers} processes...")
        with multiprocessing.Pool(self.num_workers) as pool:
            for res_fps, res_pops, res_idxs, res_smis in tqdm(
                pool.imap(_process_chunk, chunks), total=len(chunks)
            ):
                fps_list.extend(res_fps)
                pop_list.extend(res_pops)
                indices_list.extend(res_idxs)
                smiles_list.extend(res_smis)
        
       
        self.fingerprints = np.array(fps_list, dtype=np.uint8)
        self.popcounts = np.array(pop_list, dtype=np.int16) 
        self.original_indices = np.array(indices_list, dtype=np.int64)
        self.smiles_list = smiles_list
        
        
        print("Loading features...")
        if self.feature_file.endswith('.npy'):
            full_features = np.load(self.feature_file, mmap_mode='r')
        elif self.feature_file.endswith('.pt'):
            data = torch.load(self.feature_file, map_location='cpu')
            full_features = data['embedding'] if 'embedding' in data else data.numpy()
        
        self.features = np.array(full_features[self.original_indices], dtype=np.float32)
        
        # cache
        if self.cache_file:
            print("Saving cache...")
            np.savez_compressed(
                self.cache_file,
                fingerprints=self.fingerprints,
                popcounts=self.popcounts,
                features=self.features,
                indices=self.original_indices,
                smiles=np.array(self.smiles_list, dtype=object)
            )

    def _build_index(self):
        """构建 FAISS Binary Index"""
        d = self.fingerprints.shape[1] * 8 # bits, usually 2048
        print(f"Building FAISS Binary Index for {d} bits...")
        
        
        n_samples = len(self.fingerprints)
        
        if n_samples > 5000000:
            
            quantizer = faiss.IndexBinaryFlat(d)
            self.index = faiss.IndexBinaryIVF(quantizer, d, min(4096, int(np.sqrt(n_samples))))
            self.index.nprobe = 10 
            print("Training IVF index...")
            self.index.train(self.fingerprints)
        else:
            
            self.index = faiss.IndexBinaryFlat(d)
        
        self.index.add(self.fingerprints)
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_binary_cpu_to_gpu(res, 0, self.index)
                print("Index moved to GPU.")
            except AttributeError:
                print("Warning: faiss-gpu does not support binary indexes perfectly in all versions, using CPU.")

    def _smi_to_fp_and_pop(self, smiles):
        """辅助函数：单条查询转指纹和popcount"""
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        fp_obj = generator.GetFingerprint(mol)
        
        # numpy uint8
        binary_text = DataStructs.BitVectToBinaryText(fp_obj)
        arr = np.frombuffer(binary_text, dtype=np.uint8)
        
        # Popcount
        pop = fp_obj.GetNumOnBits()
        return arr.reshape(1, -1), pop

    def query(self, smiles: str, top_k: int = 1, verbose=False):
        """
        Tanimoto(A,B) = (PopA + PopB - Hamming(A,B)) / (PopA + PopB + Hamming(A,B))
        """
        t0 = time.time()
        
        # 1. 生成查询指纹
        q_fp, q_pop = self._smi_to_fp_and_pop(smiles)
        
        # 2. FAISS 搜索 (Hamming Distance)
        # 策略：为了得到精确的TopK Tanimoto，我们需要在汉明距离上多取一些样本 (Oversampling)
        # 因为 Hamming 排序 != Tanimoto 排序 (虽然强相关)
        limit_k = 2048
        multiplier = 50
        search_k = max(top_k * multiplier, limit_k)
        search_k = min(len(self.fingerprints), search_k)
        
        t1 = time.time()
        hamming_dists, indices = self.index.search(q_fp, search_k)
        t_search = time.time() - t1
        
        # 3. 计算精确 Tanimoto
        # 获取候选集的 Popcounts
        candidate_indices = indices[0]
        valid_mask = candidate_indices != -1
        candidate_indices = candidate_indices[valid_mask]
        candidate_dists = hamming_dists[0][valid_mask]
        
        candidate_pops = self.popcounts[candidate_indices] 
        
        # 向量化计算 Tanimoto
        # Intersection = (PopA + PopB - Hamming) / 2
        # Union = (PopA + PopB + Hamming) / 2
        # Tanimoto = Intersection / Union
        #          = (PopA + PopB - Hamming) / (PopA + PopB + Hamming)
        
        numerator = q_pop + candidate_pops - candidate_dists
        denominator = q_pop + candidate_pops + candidate_dists
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tanimoto_sims = numerator / denominator
            tanimoto_sims = np.nan_to_num(tanimoto_sims)
            
    
        sorted_args = np.argsort(-tanimoto_sims)
        final_top_k_args = sorted_args[:top_k]
        
        results = []
        for arg in final_top_k_args:
            idx = candidate_indices[arg]    
            sim = tanimoto_sims[arg]
            
            orig_idx = self.original_indices[idx]
            feature = self.features[idx]
            match_smiles = self.smiles_list[idx]
            
            results.append({
                'index': int(orig_idx),
                'smiles': match_smiles,
                'similarity': float(sim),
                'hamming_dist': int(candidate_dists[arg]),
                'feature': feature
            })
            
        times = {'total': time.time() - t0, 'search': t_search}
        
        if verbose:
            print(f"Search time: {times['total']*1000:.2f}ms (Core search: {times['search']*1000:.2f}ms)")
            
        return results, times
    
    def query_mean_pooling_result(self, smiles: str, top_k: int = 5, verbose: bool = False):
        """
        
        Args:
            smiles (str): 查询分子的 SMILES
            top_k (int): 取前多少个邻居进行平均 (默认5)
            verbose (bool): 是否打印日志
            
        Returns:
            dict: 包含 mean pooling 特征的字典，格式与 query 返回的单条结果一致。
                    如果查询失败或无结果，返回 None.
        """
        results, _ = self.query(smiles, top_k=top_k, verbose=verbose)
        
        if not results:
            if verbose:
                print(f"No neighbors found for {smiles}")
            return None
        
        features_stack = np.stack([item['feature'] for item in results]).astype(np.float32)
        similarities = [item['similarity'] for item in results]
        hamming_dists = [item['hamming_dist'] for item in results]
        
        
        mean_feature = np.mean(features_stack, axis=0)
        avg_similarity = float(np.mean(similarities))
        avg_hamming = int(np.mean(hamming_dists))
        
        
        result_entry = {
            'index': -11,                         # -11 just a flag
            'smiles': smiles,                    
            'similarity': avg_similarity,        
            'hamming_dist': avg_hamming,         
            'feature': mean_feature,             
            'source_count': len(results)         
        }
        
        return result_entry

    
    def batch_query(self, smiles_list: List[str], top_k: int = 10, verbose: bool = False):
        """
    
        Returns:
            List[List[dict]]
        """
        if not smiles_list:
            return []

        t0 = time.time()
        
        
        final_results = [[] for _ in smiles_list]

        
        query_fps = []
        query_pops = []
        valid_indices = []
        
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fp_obj = generator.GetFingerprint(mol)
                binary_text = DataStructs.BitVectToBinaryText(fp_obj)
                arr = np.frombuffer(binary_text, dtype=np.uint8).copy()
                
                query_fps.append(arr)
                query_pops.append(fp_obj.GetNumOnBits())
                valid_indices.append(i)
            except Exception:
                continue

        if not query_fps:
            return final_results

        query_fps = np.vstack(query_fps)
        query_pops = np.array(query_pops, dtype=np.int16)

        #  FAISS 
        limit_k = 2048
        multiplier = 50
        search_k = max(top_k * multiplier, limit_k)
        search_k = min(len(self.fingerprints), search_k)

        t1 = time.time()
        hamming_dists, indices = self.index.search(query_fps, search_k)
        search_time = time.time() - t1

        # 3.  Tanimoto 
        for j, orig_idx in enumerate(valid_indices):
            candidate_indices = indices[j]
            candidate_dists = hamming_dists[j]

            valid_mask = candidate_indices != -1
            cand_idx = candidate_indices[valid_mask]
            cand_dist = candidate_dists[valid_mask]

            if len(cand_idx) == 0:
                continue

            cand_pops = self.popcounts[cand_idx]
            q_pop = query_pops[j]

            # 向量化 Tanimoto
            numerator = q_pop + cand_pops - cand_dist
            denominator = q_pop + cand_pops + cand_dist
            
            with np.errstate(divide='ignore', invalid='ignore'):
                tanimoto = np.where(denominator > 0, numerator / denominator, 0.0)

            # Top-K
            if len(tanimoto) <= top_k:
                sorted_args = np.argsort(-tanimoto)
            else:
                
                part_idx = np.argpartition(-tanimoto, top_k)[:top_k]
                sorted_args = part_idx[np.argsort(-tanimoto[part_idx])]

            results = []
            for arg in sorted_args:
                idx = cand_idx[arg]
                results.append({
                    'index': int(self.original_indices[idx]),
                    'smiles': self.smiles_list[idx],
                    'similarity': float(tanimoto[arg]),
                    'hamming_dist': int(cand_dist[arg]),
                    'feature': self.features[idx]
                })
            
            final_results[orig_idx] = results

        if verbose:
            valid_count = len(valid_indices)
            print(f"Batch query: {valid_count}/{len(smiles_list)} valid, "
                f"{time.time() - t0:.3f}s total (FAISS: {search_time:.3f}s)")

        return final_results
    def batch_get_mean_pooling_result(self, smiles_list: List[str], top_k: int = 5, verbose: bool = False):
        """
        
        Args:
            smiles_list (List[str])
            top_k (int):
            verbose (bool)
            
        Returns:
            List[dict | None]: 
                                - success: dict  'feature' (mean pooled) 
                                - failed: None。
        """
        batch_neighbors = self.batch_query(smiles_list, top_k=top_k, verbose=verbose)
        
        mean_results = []
        
        for i, neighbors in enumerate(batch_neighbors):
            if not neighbors:
                mean_results.append(None)
                continue
            
            features_stack = np.stack([item['feature'] for item in neighbors]).astype(np.float32)
            similarities = [item['similarity'] for item in neighbors]
            hamming_dists = [item['hamming_dist'] for item in neighbors]
            
            # (Mean Pooling)
            mean_feature = np.mean(features_stack, axis=0)
            avg_sim = float(np.mean(similarities))   
            avg_ham = int(np.mean(hamming_dists))    
            
            
            result_entry = {
                'index': -11,                         # -11 just a flag
                'smiles': smiles_list[i],            #  SMILES
                'similarity': avg_sim,               
                'hamming_dist': avg_ham,             
                'feature': mean_feature,             
                'source_count': len(neighbors)       
            }
            
            mean_results.append(result_entry)
            
        return mean_results


# -----------------------------------------------------------------------------
# for test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    entity_file = 'Data/Train/for_embedding/all_molecules_clean.txt'
    features_path = 'rgcn/global_emb_FP_512/model_epoch197_0.00307_embedding.npy'
    ckpt_dir = "models"
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_file = f"{ckpt_dir}/fp_cache.npz"
    
    searcher = SimilaritySearcher(entity_file, features_path, cache_file, use_gpu=False)

    test_smiles = 'CCOC(=O)c1cc2c(F)cccc2nc1[C@H](C)NC(=O)OC(C)(C)C'
    try:
        results, _ = searcher.query(test_smiles, top_k=1, verbose=True)
        
        print(f"\nQuery: {test_smiles}")
        print("-" * 50)
        for i, res in enumerate(results):
            print(f"Rank {i+1}: Sim={res['similarity']:.4f} | Hamming={res['hamming_dist']} | ID={res['index']}")
            print(f"        SMILES: {res['smiles']}")
    except Exception as e:
        print(f"Error: {e}")