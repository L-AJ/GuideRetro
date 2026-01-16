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
# for multiprocessing
# -----------------------------------------------------------------------------
def _process_chunk(chunk_data):
    """
    处理一个数据块：解析SMILES -> 生成指纹 -> 计算Popcount
    """
    indices, smiles_list = chunk_data
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    fps = []
    popcounts = [] # 存储指纹中'1'的个数，用于精确计算Tanimoto
    valid_indices = []
    valid_smiles = []
    
    for idx, smi in zip(indices, smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            # 生成指纹
            fp_obj = generator.GetFingerprint(mol)
            
            # 1. 转换为二进制 bytes (非常快且省内存)
            binary_text = DataStructs.BitVectToBinaryText(fp_obj)
            fp_arr = np.frombuffer(binary_text, dtype=np.uint8)
            
            # 2. 计算 Popcount (位为1的个数)
            # 这一步对 Tanimoto 精确计算至关重要
            pop = fp_obj.GetNumOnBits()
            
            fps.append(fp_arr)
            popcounts.append(pop)
            valid_indices.append(idx)
            valid_smiles.append(smi)
            
        except Exception:
            continue
            
    return fps, popcounts, valid_indices, valid_smiles

# -----------------------------------------------------------------------------
# main class
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
        self.popcounts = None       # int16 array (N,) - 预计算的位数
        self.features = None        # float32 array (N, D)
        self.smiles_list = None     # List[str]
        self.original_indices = None # 映射回原始文件的行号
        
        self._load_data()
        self._build_index()

    def _load_data(self):
        """加载数据（优先缓存，其次多进程处理）"""
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
        
        # 1. 读取所有文本行
        raw_entries = []
        with open(self.entity_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # 假设格式: ID \t SMILES
                    raw_entries.append((int(parts[0]), parts[1]))
        
        total_items = len(raw_entries)
        print(f"Total raw entries: {total_items}")
        
        # 2. 准备分块数据进行多进程处理
        chunk_size = max(100, total_items // (self.num_workers * 4))
        chunks = []
        for i in range(0, total_items, chunk_size):
            batch = raw_entries[i:i + chunk_size]
            indices = [x[0] for x in batch]
            smiles = [x[1] for x in batch]
            chunks.append((indices, smiles))
            
        # 3. 多进程处理
        fps_list = []
        pop_list = []
        indices_list = []
        smiles_list = []
        
        print(f"Processing fingerprints with {self.num_workers} processes...")
        with multiprocessing.Pool(self.num_workers) as pool:
            # 使用 imap 进行有序返回及显示进度条
            for res_fps, res_pops, res_idxs, res_smis in tqdm(
                pool.imap(_process_chunk, chunks), total=len(chunks)
            ):
                fps_list.extend(res_fps)
                pop_list.extend(res_pops)
                indices_list.extend(res_idxs)
                smiles_list.extend(res_smis)
        
        # 4. 转换格式
        self.fingerprints = np.array(fps_list, dtype=np.uint8)
        self.popcounts = np.array(pop_list, dtype=np.int16) # Popcount不会超过2048，int16够用
        self.original_indices = np.array(indices_list, dtype=np.int64)
        self.smiles_list = smiles_list
        
        # 5. 加载特征 (只加载有效指纹对应的特征)
        print("Loading features...")
        if self.feature_file.endswith('.npy'):
            full_features = np.load(self.feature_file, mmap_mode='r')
        elif self.feature_file.endswith('.pt'):
            data = torch.load(self.feature_file, map_location='cpu')
            full_features = data['embedding'] if 'embedding' in data else data.numpy()
        
        # 根据原始索引提取特征
        self.features = np.array(full_features[self.original_indices], dtype=np.float32)
        
        # 6. 保存缓存
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
        
        # 对于百万级数据，BinaryFlat (暴力搜索) 在现代CPU/GPU上非常快
        # 如果数据量超过 500万，建议切换到 IndexBinaryIVF
        n_samples = len(self.fingerprints)
        
        if n_samples > 5000000:
            # 大规模数据使用倒排索引
            quantizer = faiss.IndexBinaryFlat(d)
            self.index = faiss.IndexBinaryIVF(quantizer, d, min(4096, int(np.sqrt(n_samples))))
            self.index.nprobe = 10 # 搜索时的探测簇数
            print("Training IVF index...")
            self.index.train(self.fingerprints)
        else:
            # 百万级以下直接用Flat，最精确且不需要训练
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
        
        # 转numpy uint8
        binary_text = DataStructs.BitVectToBinaryText(fp_obj)
        arr = np.frombuffer(binary_text, dtype=np.uint8)
        
        # Popcount
        pop = fp_obj.GetNumOnBits()
        return arr.reshape(1, -1), pop

    def query(self, smiles: str, top_k: int = 1, verbose=False):
        """
        查询并返回精确的 Tanimoto 相似度
        原理：Tanimoto(A,B) = (PopA + PopB - Hamming(A,B)) / (PopA + PopB + Hamming(A,B))
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
        candidate_indices = indices[0]
        valid_mask = candidate_indices != -1
        candidate_indices = candidate_indices[valid_mask]
        candidate_dists = hamming_dists[0][valid_mask]
        
        candidate_pops = self.popcounts[candidate_indices] 
        
        numerator = q_pop + candidate_pops - candidate_dists
        denominator = q_pop + candidate_pops + candidate_dists
        
        # 避免除以0 (虽然理论上不可能，除非全0指纹且hamming为0)
        with np.errstate(divide='ignore', invalid='ignore'):
            tanimoto_sims = numerator / denominator
            tanimoto_sims = np.nan_to_num(tanimoto_sims)
            
        # 4. 重新排序，取真正的 Top K
        # np.argsort 是升序，我们需要降序
        sorted_args = np.argsort(-tanimoto_sims)
        final_top_k_args = sorted_args[:top_k]
        
        results = []
        for arg in final_top_k_args:
            idx = candidate_indices[arg]    # 数据库中的内部索引
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
    
    def batch_query(self, smiles_list: List[str], top_k: int = 10, verbose: bool = False):
        """
        批量查询多个 SMILES，返回每个查询的 Top-K 结果
        
        Returns:
            List[List[dict]]: 与输入 smiles_list 一一对应的结果
        """
        if not smiles_list:
            return []

        t0 = time.time()
        
        # 预分配结果
        final_results = [[] for _ in smiles_list]

        # 1. 批量生成查询指纹
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

        # 2. FAISS 批量搜索
        limit_k = 2048
        multiplier = 50
        search_k = max(top_k * multiplier, limit_k)
        search_k = min(len(self.fingerprints), search_k)

        t1 = time.time()
        hamming_dists, indices = self.index.search(query_fps, search_k)
        search_time = time.time() - t1

        # 3. 批量精确 Tanimoto 重排序
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

            # 取 Top-K
            if len(tanimoto) <= top_k:
                sorted_args = np.argsort(-tanimoto)
            else:
                # 使用 argpartition 更快
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

    def get_random_result(self, seed: int = None):
        """
        随机从数据库中获取一个分子的特征，作为完全无关的参考。
        
        Args:
            seed (int, optional): 随机种子。如果传入固定整数，每次调用将返回相同的结果（用于复现）。
                                  如果不传 (None)，则完全随机。
        """
        import random
        
        # 1. 实例化一个独立的随机生成器，避免影响全局的 random.seed()
        # 如果 seed 为 None，Random 会使用系统时间作为种子
        rng = random.Random(seed)
        
        # 2. 检查数据是否为空
        n_total = len(self.features)
        if n_total == 0:
            return None

        # 3. 生成随机索引
        idx = rng.randint(0, n_total - 1)
        
        # 4. 返回结果
        return {
            'index': int(self.original_indices[idx]),
            'smiles': self.smiles_list[idx],
            'similarity': 0.0, 
            'hamming_dist': -1,
            'feature': self.features[idx]
        }
# -----------------------------------------------------------------------------
# 测试代码
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    entity_file = 'Data/Train/for_embedding/all_molecules_clean.txt'
    features_path = 'rgcn/global_emb_FP_512/model_epoch197_0.00307_embedding.npy'
    ckpt_dir = "FusionRetro/models_final_FP512_Noclip_moresize"
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_file = f"{ckpt_dir}/fp_cache.npz"
    
    searcher = SimilaritySearcher(entity_file, features_path, cache_file, use_gpu=False)
    
    # 测试查询
    # test_smiles = "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)(C(=O)CO)CC3OC1CC(NC(=O)C(F)(F)C(F)(F)F)C(O)C(C)O1" #0.75
    # test_smiles = "CC(C)C(ON=C(C(=O)NC1C(=O)N(S(=O)(=O)O)C1COC(N)=O)c1csc(N)n1)c1cc(=O)c(O)cn1O"  #0.44
    # test_smiles = "CC(CC(=O)C(C)C(C)C)C1=C(O)C(=O)C2C3=C(CCC12C)C1(C)CCC(OC2OC(C(=O)O)C(O)C(OC4OCC(O)C(O)C4O)C2O)CC1CC3" #0.24
    # test_smiles = 'FC(F)(F)Cn1ncnc1-c1cc2n(n1)-c1cc(C3CCNCC3)ccc1OCC2' #Rank 1: Sim=0.6970 | Hamming=20 | ID=895812 SMILES: FC(F)(F)Cn1ncnc1-c1cc2n(n1)-c1cc(Br)ccc1OCC2
    # test_smiles = 'CC1(C)OCC(C)(CO)N(Cc2ccccc2)C1=O' #Rank 1: Sim=0.6364 | Hamming=16 | ID=220122 SMILES: CC1(CO)COCC(=O)N1Cc1ccccc1
    test_smiles='CCOC(=O)c1cc2c(F)cccc2nc1[C@H](C)NC(=O)OC(C)(C)C'
    try:
        results, _ = searcher.query(test_smiles, top_k=5, verbose=True)
        
        print(f"\nQuery: {test_smiles}")
        print("-" * 50)
        for i, res in enumerate(results):
            print(f"Rank {i+1}: Sim={res['similarity']:.4f} | Hamming={res['hamming_dist']} | ID={res['index']}")
            print(f"        SMILES: {res['smiles']}")
    except Exception as e:
        print(f"Error: {e}")