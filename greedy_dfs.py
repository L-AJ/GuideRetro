import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import random
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from rdkit import Chem
from Similar_search import SimilaritySearcher
from functools import lru_cache
import concurrent.futures

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')

char_to_ix = None
ix_to_char = None
vocab_size = None
device = None


@lru_cache(maxsize=100000)
def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles

@lru_cache(maxsize=100000)
def get_inchikey_prefix(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)[:14]
    except:
        return None

def check_reactant_is_material(reactant):
    key = get_inchikey_prefix(reactant)
    return key is not None and key in stock_inchikeys

def check_reactants_are_material(reactants):
    for reactant in reactants:
        key = get_inchikey_prefix(reactant)
        if key is None or key not in stock_inchikeys:
            return False
    return True


def prepare_encoder_inputs(products_list, max_depth, max_length):
    batch_size = len(products_list)
    products_input = torch.zeros((batch_size, max_depth, max_length), device=device, dtype=torch.long)
    products_input_mask = torch.zeros((batch_size, max_depth, max_length), device=device)
    
    for index, products in enumerate(products_list):
        for i, product in enumerate(products):
            product = '^' + product + '$'
            seq_ids = [char_to_ix[s] for s in product]
            length = len(seq_ids)
            products_input[index, i, :length] = torch.tensor(seq_ids, device=device)
            products_input_mask[index, i, :length] = 1
            
    return products_input, products_input_mask

def prepare_decoder_inputs_fast(products_list, candidates, max_depth, max_length):
    batch_size = len(candidates)
    
    reactants_input = torch.zeros((batch_size, max_depth, max_length), device=device, dtype=torch.long)
    reactants_input_mask = torch.zeros((batch_size, max_depth, max_length), device=device)
    memory_input_mask = torch.zeros((batch_size, max_depth), device=device)

    
    if len(products_list) > 0:
        real_num_products = len(products_list[0])
        memory_input_mask[:, :real_num_products] = 1
    

    for index, reactants in enumerate(candidates):
        for i, reactant in enumerate(reactants):
            reactant = '^' + reactant + '$'
            seq_ids = [char_to_ix[s] for s in reactant[:-1]]
            length = len(seq_ids)
            reactants_input[index, i, :length] = torch.tensor(seq_ids, device=device)
            reactants_input_mask[index, i, :length] = 1
            
    return reactants_input, reactants_input_mask, memory_input_mask

def get_output_probs_vectorized(products_tensors, candidates, products_len, max_depth, max_length, product_exter_feature):
    batch_size = len(candidates)
    
    # 1. Expand Encoder Inputs
    p_ids_base, p_mask_base = products_tensors
    products_ids = p_ids_base.expand(batch_size, -1, -1)
    products_mask = p_mask_base.expand(batch_size, -1, -1)

    # 2. Prepare Decoder Inputs
    dummy_products_list = [range(products_len)] * batch_size 
    reactants_ids, reactants_mask, memory_mask = prepare_decoder_inputs_fast(
        dummy_products_list, candidates, max_depth, max_length
    )

    # 3. Expand Features
    if isinstance(product_exter_feature, np.ndarray):
        product_exter_feature = torch.from_numpy(product_exter_feature).float()
    product_exter_feature = product_exter_feature.to(device)
    
    if product_exter_feature.dim() == 2:
        expanded_features = product_exter_feature.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        expanded_features = product_exter_feature.expand(batch_size, -1, -1)

    # 4. Masks
    mutual_mask = get_mutual_mask([reactants_mask, products_mask])
    products_mask = get_padding_mask(products_mask)
    reactants_mask = get_tril_mask(reactants_mask)
    memory_mask = get_mem_tril_mask(memory_mask)

    # 5. Inference
    logits = predict_model(
        products_ids, reactants_ids, products_mask, reactants_mask, 
        mutual_mask, memory_mask, expanded_features
    )

    # 6. Extract Probs
    k = products_len - 1
    target_indices = torch.tensor([len(c[k]) for c in candidates], device=device)
    relevant_logits = logits[:, k, :, :] 
    batch_indices = torch.arange(batch_size, device=device)
    target_logits = relevant_logits[batch_indices, target_indices, :]
    
    target_logits = target_logits / args.temperature
    probs = F.softmax(target_logits, dim=-1)
    
    return probs

def get_beam(products, product_exter_feature, beam_size, length_penalty_alpha=1):
    p_input, p_mask = prepare_encoder_inputs([products], len(products), args.max_length)
    products_tensors = (p_input, p_mask)
    
    lines = [""] * beam_size
    scores = torch.zeros(beam_size, device=device)
    final_beams = []
    object_size = beam_size
    base_res = products[:-1]

    for step in range(args.max_length):
        if len(lines) == 0:
            break

        if step == 0:
            current_batch_lines = [""]
            current_scores = torch.zeros(1, device=device)
        else:
            current_batch_lines = lines
            current_scores = scores

        candidates_res = [base_res + [line] for line in current_batch_lines]

        probs = get_output_probs_vectorized(
            products_tensors, candidates_res, len(products), len(products), 
            args.max_length, product_exter_feature
        )

        vocab_probs = probs
        
        # Calculate scores: score - log(prob)
        total_scores = current_scores.unsqueeze(1) - torch.log10(vocab_probs + 1e-10)

        flat_scores = total_scores.view(-1)
        k_candidates = min(flat_scores.shape[0], object_size * 2 ) 
        topk_scores, topk_indices = torch.topk(flat_scores, k=k_candidates, largest=False)

        topk_scores = topk_scores.detach().cpu().numpy()
        topk_indices = topk_indices.detach().cpu().numpy()
        
        new_lines = []
        new_scores = []
        
        batch_indices = topk_indices // vocab_size
        char_indices = topk_indices % vocab_size
        
        for i in range(len(topk_indices)):
            if len(new_lines) >= object_size:
                break    
            
            batch_idx = batch_indices[i]
            char_idx = char_indices[i]
            score = topk_scores[i]
            symbol = ix_to_char[char_idx]
            
            prev_line = "" if step == 0 else lines[batch_idx]

            if symbol == '$':
                added = prev_line + symbol
                if added != "$":
                    final_beams.append([added, score])
                object_size -= 1
            else:
                new_lines.append(prev_line + symbol)
                new_scores.append(score)
        
        lines = new_lines
        scores = torch.tensor(new_scores, device=device)
        
        if object_size <= 0:
            break

    # Length Penalty
    for i in range(len(final_beams)):
        length = len(final_beams[i][0])
        penalty = ((5 + length) / 6) ** length_penalty_alpha
        final_beams[i][1] = final_beams[i][1] / penalty
    

    final_beams = list(sorted(final_beams, key=lambda x:x[1]))
    
    answer = []
    aim_size = beam_size
    
    seen_results = set()
    
    for k in range(len(final_beams)):
        if aim_size == 0:
            break
            
    
        raw_str = final_beams[k][0].replace("^", "").replace("$", "").strip()
        reactant_strs = set(raw_str.split("."))
        
        valid_reactants_cano = []
        all_valid = True
        
        for r in reactant_strs:
            if not r: continue
            
            try:
                m = Chem.MolFromSmiles(r)
                if m is None:
                    all_valid = False
                    break
                
                for atom in m.GetAtoms():
                    atom.ClearProp('molAtomMapNumber')
                m = Chem.RemoveHs(m)
                smi = Chem.MolToSmiles(m, isomericSmiles=True)
                if not smi:
                    all_valid = False
                    break
                valid_reactants_cano.append(smi)
            except:
                all_valid = False
                break
        
        if not all_valid or len(valid_reactants_cano) == 0:
            continue
       
        valid_reactants_cano.sort()
        res_tuple = tuple(valid_reactants_cano)
        
        if res_tuple not in seen_results:
            seen_results.add(res_tuple)
            answer.append([list(valid_reactants_cano), final_beams[k][1]])
            aim_size -= 1
            
    return answer

# ==================== (Greedy DFS) ====================

def get_route_result(task, searcher=None):
    with torch.inference_mode():
        max_depth = task["depth"]
        results= searcher.query_mean_pooling_result(task["product"], top_k=5)
        product_exter_feature = results['feature']
        
        if isinstance(product_exter_feature, np.ndarray):
            product_exter_feature = product_exter_feature.reshape(1, -1)
        
        queue = []
        queue.append({
            "routes_info": [{
                "route": [task["product"]],
                "depth": 0,
                "product_exter_feature": product_exter_feature
            }],
            "starting_materials": [],
        })
        
        # Greedy Search
        while len(queue) > 0:
            curr_state = queue.pop(0) 
            
            routes_info = curr_state["routes_info"]
            starting_materials = curr_state["starting_materials"]
            
            if len(routes_info) == 0:
                # Check Match
                answer_keys = set()
                for m in starting_materials:
                    key = get_inchikey_prefix(m)
                    if key: answer_keys.add(key)
                
                ground_truth_keys_list = task["targets"] 
                
                for gt in ground_truth_keys_list:
                    if gt == answer_keys:
                        return max_depth, True
                return max_depth, False

            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            product_exter_feature = first_route_info["product_exter_feature"]
            
            if depth > max_depth * 2:
                break
            
            beam_results = get_beam(first_route, product_exter_feature, args.beam_size, args.length_penalty_alpha)
            
            if len(beam_results) == 0:
                break

            # Greedy: Top-1
            expansion_solution = beam_results[0]
            expansion_reactants, _ = expansion_solution[0], expansion_solution[1]
            expansion_reactants = sorted(expansion_reactants)
            
            iter_routes = deepcopy(routes_info)
            iter_routes.pop(0)
            iter_starting_materials = list(starting_materials)
            
            if check_reactants_are_material(expansion_reactants):
                iter_starting_materials.extend(expansion_reactants)
                queue.append({
                    "routes_info": iter_routes,
                    "starting_materials": iter_starting_materials
                })
            else:
                new_task_routes = []
                for reactant in expansion_reactants:
                    if check_reactant_is_material(reactant):
                        iter_starting_materials.append(reactant)
                    else:
                        results = searcher.query_mean_pooling_result(reactant, top_k=5)
                        add_feature = results['feature']
                        if isinstance(add_feature, np.ndarray):
                            add_feature = add_feature.reshape(1, -1)
                            
                        if isinstance(product_exter_feature, torch.Tensor):
                            # === 核心修复：detach() ===
                            pf_cpu = product_exter_feature.detach().cpu().numpy()
                        else:
                            pf_cpu = product_exter_feature
                            
                        new_product_exter_feature = np.vstack([pf_cpu, add_feature])
                        
                        new_task_routes.append({
                            "route": first_route + [reactant],
                            "depth": depth + 1,
                            "product_exter_feature": new_product_exter_feature
                        })
                
                iter_routes = new_task_routes[::-1] + iter_routes
                queue.append({
                    "routes_info": iter_routes,
                    "starting_materials": iter_starting_materials
                })

        return max_depth, False

def load_dataset(split, data_dir=""):
    file_name = os.path.join(data_dir, f"{split}_dataset.json")
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return []
        
    dataset = [] 
    total_gt_sets_count = 0  
    
    count_valid_products = 0     
    count_no_materials = 0      
    
    print(f"Loading {split} dataset from {file_name}...")
    
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        
        
        for _, reaction_trees in tqdm(_dataset.items(), desc="Processing Data"):
            try:
                
                if 'retro_routes' in reaction_trees.get('1', {}) and len(reaction_trees['1']['retro_routes']) > 0:
                    raw_prod = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
                else:
                    continue 
                
                mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(raw_prod)))
                if mol: 
                    product_smi = Chem.MolToSmiles(mol)
                else:
                    product_smi = raw_prod
                
                _, product = cano_smiles(product_smi)
                if product is None: continue # 如果产物都不合法，跳过该条目
                
                count_valid_products += 1 # 产物合法，计数+1

            except Exception:
                continue

        
            valid_gt_sets = [] 
            
            num_trees = int(reaction_trees.get('num_reaction_trees', 0))
            
            for i in range(1, num_trees + 1):
                tree_key = str(i)
                if tree_key not in reaction_trees: continue
                
                materials = reaction_trees[tree_key].get('materials', [])
                
                current_set = set()
                is_route_valid = True
                
                for mat in materials:
                    key = get_inchikey_prefix(mat)
                    if key is None:
                        is_route_valid = False
                        break
                    current_set.add(key)
            
                if is_route_valid and len(current_set) > 0:
                    valid_gt_sets.append(current_set)
            
            if valid_gt_sets:
                dataset.append({
                    "product": product, 
                    "targets": valid_gt_sets, 
                    "depth": reaction_trees.get('depth', 0)
                })
                total_gt_sets_count += len(valid_gt_sets)
            else:
                # 这是一个“有产物但无原料”的孤儿数据
                count_no_materials += 1

    
    print(f"\nDataset '{split}' loaded successfully.")
    print(f"Total valid products processed: {count_valid_products}")
    print(f"--------------------------------------------------")
    print(f"1. Molecules KEPT (have >0 valid routes): {len(dataset)}")
    print(f"2. Molecules DROPPED (have 0 valid routes): {count_no_materials}")
    
    
    if count_valid_products > 0:
        coverage = (len(dataset) / count_valid_products) * 100
        print(f"   -> Coverage (Kept / Valid Products): {coverage:.2f}%")
    
    if count_no_materials == 0:
        print("   -> [PERFECT] Every valid product has at least one starting material set.")
    else:
        print(f"   -> [WARNING] {count_no_materials} molecules were skipped because no valid starting materials were found.")
        
    print(f"--------------------------------------------------")
    print(f"Total valid ground truth sets (routes): {total_gt_sets_count}")
    print(f"Average routes per molecule (in final dataset): {total_gt_sets_count / len(dataset) if dataset else 0:.2f}\n")
    
    return dataset

# ==================== mitiprocess ====================

def run_parallel_greedy_ordered(tasks, searcher, log_path, max_workers=4):
    final_depth_stats = np.zeros((2, 16)) # [hit, total] per depth
    
    next_to_write = 0
    pending_results = {} # 缓冲池 {idx: (max_depth, match)}
   
    seq_hit = 0
    seq_total = 0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{'Time':19}  {'Idx':>4}  {'Depth':>5}  {'Hit':>6}  {'CurrAcc':>8}\n")

    print(f"\n Greedy DFS  (Ordered Logging)， {len(tasks)} tasks (Workers={max_workers})")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(get_route_result, task, searcher): idx 
            for idx, task in enumerate(tasks)
        }
        
        pbar = tqdm(total=len(tasks), desc="Processing", unit="mol")
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                max_depth, match = future.result()
                res = (max_depth, match)
            except Exception as e:
                print(f"Task {idx+1} failed: {e}")
                res = (0, False) # 失败默认值
            
            
            pending_results[idx] = res
            
            while next_to_write in pending_results:
                d_depth, d_match = pending_results.pop(next_to_write)
                
                
                seq_total += 1
                if d_match:
                    seq_hit += 1
                
                if d_depth < 16:
                    final_depth_stats[1, d_depth] += 1
                    if d_match:
                        final_depth_stats[0, d_depth] += 1

                now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                hit_str = "Yes" if d_match else "No"
                curr_acc = 100.0 * seq_hit / (seq_total + 1e-8)
                
                log_line = f"{now}  {next_to_write+1:4d}  {d_depth:5d}  {hit_str:>6}  {curr_acc:6.2f}%"
                
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
                
               
                pbar.set_postfix_str(f"LogIdx={next_to_write+1} Acc={curr_acc:.2f}%")
               
                next_to_write += 1
               
                pbar.update(1)
    
    overall_result = np.array([seq_hit, seq_total])
    return overall_result, final_depth_stats

# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=14)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=640)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=10)
    parser.add_argument('--intermediate_size', type=int, default=512)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.5)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--length_penalty_alpha', type=float, default=1)
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Init
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    config = TransformerConfig(vocab_size=get_vocab_size(),
                            max_length=args.max_length,
                            embedding_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=args.num_attention_heads,
                            intermediate_size=args.intermediate_size,
                            hidden_dropout_prob=args.hidden_dropout_prob)

    predict_model = Transformer(config)
    
    # Paths
    entity_file = 'Data/Train/for_embedding/all_molecules_clean.txt'
    features_path = 'rgcn/global_emb_FP_512/embedding.npy'
    ckpt_dir = "models"
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_file = ckpt_dir + "/fp_cache.npz"

    checkpoint_path = "models/finetune_best_model.pth"

    # Load Model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    predict_model.load_state_dict(state_dict, strict=True)
    predict_model.to(device)
    predict_model.eval()

    # Load Data
    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()
    
    print("Loading stock...")
    stock = pd.read_hdf('Data/zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])
    
    print("Loading tasks...")
    tasks = load_dataset('test', data_dir="Data/Test")
    
    print("Initializing Searcher...")
    searcher = SimilaritySearcher(entity_file, features_path, cache_file, use_gpu=False)

    log_path = f"result_greedy_B{args.beam_size}_T{args.temperature}_P{args.length_penalty_alpha}_random.txt"
    
    # Execution
    with torch.inference_mode():
        overall_result, depth_hit = run_parallel_greedy_ordered(
            tasks, 
            searcher=searcher, 
            log_path=log_path, 
            max_workers=args.workers
        )

    print("\nFinal Result:")
    acc = 100 * overall_result[0] / max(overall_result[1], 1)  
    print(f"Overall Acc: {acc:.2f}% ({int(overall_result[0])}/{int(overall_result[1])})")

    # caculate depth hit percent
    depth_hit_percent = np.where(
        depth_hit[1, :] > 0,
        100 * depth_hit[0, :] / depth_hit[1, :],
        0.0
    )

    with open(log_path, "a") as f:
        f.write("\n====== Summary ======\n")
        f.write(f"Overall Acc: {acc:.2f}%\n")
        f.write(f"Depth Hit: {depth_hit.tolist()}\n")
        f.write(f"Depth Hit Percent: {depth_hit_percent.tolist()}\n")
        f.write(f"{checkpoint_path}\n{features_path}\n{args.temperature}\n\n")
