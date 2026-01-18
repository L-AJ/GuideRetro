import argparse
import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import heapq
import datetime
import concurrent.futures
from copy import deepcopy 
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import traceback
from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from Similar_search import SimilaritySearcher
from functools import lru_cache
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')

# 1. ValueMLP
class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))
        return x

value_cache = {}
def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return np.zeros(fp_dim, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)
    return fp

def value_fn(smi):
    """
    if args.use_value = False,value= 0.0。
    """
    
    if not args.use_value:
        return 0.0
    
    if smi in value_cache:
        return value_cache[smi]
    
   
    try:
        fp = smiles_to_fp(smi, fp_dim=args.fp_dim).reshape(1, -1)
        fp_tensor = torch.from_numpy(fp).float().to(device)
        
        with torch.inference_mode():
            v = value_model(fp_tensor).item()
    except:
        v = 0.0 
    
    
    if len(value_cache) > 200000:
        value_cache.clear()
        
    value_cache[smi] = v
    return v
# ==========================================
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
    with torch.inference_mode():
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
    object_size = beam_size*2
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
        k_candidates = min(flat_scores.shape[0], object_size)
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

# ==========================================
# 4. A* seach
# ==========================================
def get_route_result(task, searcher=None, idx=None):
    """
    tas
    searcher
    idx
    
    Returns:
        max_depth (int)
        rank (int or None)
        log_msgs (list)
    """
    log_msgs = []
    
    max_depth = task["depth"]
    product = task["product"]
    
    task_tag = f"Task-{idx+1}" if idx is not None else f"Task-{product[:15]}"
    
    
    try:
        results= searcher.query_mean_pooling_result(task["product"], top_k=5)
        product_exter_feature = results['feature']
        if isinstance(product_exter_feature, np.ndarray):
            product_exter_feature = product_exter_feature.reshape(1, -1)
    except Exception as e:
        log_msgs.append(f"[{task_tag}] FAIL-1 Feature query: depth={max_depth}, error={e}")
        return max_depth, None, log_msgs

    answer_set = []
    queue = []
    
    queue.append({
        "score": 0.0,
        "routes_info": [{
            "route": [task["product"]],
            "depth": 0,
            "product_exter_feature": product_exter_feature
        }],
        "starting_materials": [],
    })
    
    beam_size = args.beam_size
    iteration = 0
    total_beam_calls = 0
    total_beam_empty = 0

    # ====================== A* loop ======================
    while len(queue) > 0:
        iteration += 1
        if iteration > 1000:
            log_msgs.append(f"[{task_tag}] FAIL-2 Too many iterations: depth={max_depth}, iters={iteration}")
            break
            
        nxt_queue = []
        
        for item in queue:
            score = item["score"] 
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            
            first_route_info = routes_info[0]
            first_route = first_route_info["route"]
            depth = first_route_info["depth"]
            product_exter_feature = first_route_info["product_exter_feature"]
            expansion_mol = first_route[-1]
            
            if depth > max_depth:
                continue
            
            try:
                current_mol_value = value_fn(expansion_mol)
            except Exception as e:
                log_msgs.append(f"[{task_tag}] FAIL-3 value_fn: depth={max_depth}, error={e}")
                continue

            try:
                total_beam_calls += 1
                expansion_solutions = get_beam(
                    first_route, 
                    product_exter_feature, 
                    args.beam_size, 
                    args.length_penalty_alpha
                )
                
                if len(expansion_solutions) == 0:
                    total_beam_empty += 1
                    
            except Exception as e:
                log_msgs.append(f"[{task_tag}] FAIL-4 get_beam: depth={max_depth}, route_len={len(first_route)}, error={e}")

                continue

            for expansion_solution in expansion_solutions:
                reactants, reaction_cost = expansion_solution[0], expansion_solution[1]
                reactants = sorted(reactants)
                
                try:
                    iter_routes = routes_info[:]
                    iter_routes.pop(0) 
                    iter_starting_materials = starting_materials[:]
                    
                    all_materials = True
                    estimation_cost = 0.0 
                    new_sub_routes = [] 
                    
                    for reactant in reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            all_materials = False
                            estimation_cost += value_fn(reactant)
                            
                            try:
                            
                                results = searcher.query_mean_pooling_result(reactant, top_k=5)
                                add_feature = results['feature']
                                if isinstance(add_feature, np.ndarray):
                                    add_feature = add_feature.reshape(1, -1)
                                
                                new_sub_routes.append({
                                    "reactant": reactant,
                                    "raw_feature": add_feature
                                })
                            except Exception as e:
                                log_msgs.append(f"[{task_tag}] FAIL-5 query reactant: {reactant[:30]}, error={e}")
                                raise
                    
                    new_total_score = score - current_mol_value + reaction_cost + estimation_cost

                    if all_materials and len(iter_routes) == 0:
                        answer_set.append({
                            "score": new_total_score,
                            "starting_materials": iter_starting_materials,
                        })
                    else:
                        final_new_routes = []
                        for sub in new_sub_routes:
                            try:
                                new_feat = np.vstack([product_exter_feature, sub["raw_feature"]])
                            except Exception as e:
                                log_msgs.append(f"[{task_tag}] FAIL-6 vstack error: {e}")
                                raise
                                
                            final_new_routes.append({
                                "route": first_route + [sub["reactant"]],
                                "depth": depth + 1,
                                "product_exter_feature": new_feat
                            })
                        
                        updated_routes_info = final_new_routes[::-1] + iter_routes
                        nxt_queue.append({
                            "score": new_total_score,
                            "routes_info": updated_routes_info,
                            "starting_materials": iter_starting_materials
                        })
                        
                except Exception:
                    continue
        
        if len(nxt_queue) > beam_size:
            queue = heapq.nsmallest(beam_size, nxt_queue, key=lambda x: x["score"])
        else:
            queue = nxt_queue
    
    if len(answer_set) == 0:
        log_msgs.append(f"[{task_tag}] FAIL-7 No answers: depth={max_depth}, iters={iteration}, beam_calls={total_beam_calls}, beam_empty={total_beam_empty}")

    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []

    for item in answer_set:
        start_mats = item["starting_materials"]
        keys = set()
        valid_path = True
        
        for m in start_mats:
            k = get_inchikey_prefix(m)
            if k is None:
                valid_path = False
                break
            keys.add(k)
            
        if not valid_path or len(keys) == 0:
            continue
        
        key_frozen = frozenset(keys)
        
        if key_frozen not in record_answers:
            record_answers.add(key_frozen)
            final_answer_set.append({
                "score": item["score"],
                "answer_keys": keys
            })
            
        if len(final_answer_set) >= args.beam_size:
            break

    if len(answer_set) > 0 and len(final_answer_set) == 0:
        log_msgs.append(f"[{task_tag}] FAIL-8 All filtered out: depth={max_depth}, raw_answers={len(answer_set)}")

    # ====================== match Ground Truth ======================
    ground_truth_keys_list = task["targets"] 
    
    for rank, answer in enumerate(final_answer_set):
        answer_keys = answer["answer_keys"]
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                log_msgs.append(f"[{task_tag}] SUCCESS: iters={iteration}, beam_calls={total_beam_calls}, total_answers={len(final_answer_set)}")
                return max_depth, rank, log_msgs

    
    if len(final_answer_set) > 0:
        log_msgs.append(f"[{task_tag}] FAIL-9 No GT match: depth={max_depth}, preds={len(final_answer_set)}, gt={len(ground_truth_keys_list)}")

    return max_depth, None, log_msgs

def run_parallel_with_logging(tasks, searcher, max_workers=8, log_path="retro_search_log.txt"):
    # results：(depth, rank, log_msgs)
    results = [None] * len(tasks)
    overall_hit = np.zeros(args.beam_size, dtype=int)
    overall_total = 0
    pending_results = {}
    next_to_write = 0
    
    with open(log_path, "w", encoding="utf-8") as f:
        header = f"{'Time':19}  {'Idx':>4}  {'Depth':>5}  {'Rank':>6}  " + \
                 "  ".join([f"Top-{k+1:>8}" for k in range(args.beam_size)])
        f.write(header + "\n")
    
    print(f"\n total {len(tasks)} tasks  (Workers={max_workers})")
    print(f"log path: {log_path}\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(get_route_result, task, searcher, i): (i, task) 
            for i, task in enumerate(tasks)
        }
        
        pbar = tqdm(total=len(tasks), desc="Retro Search", unit="mol",
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for future in concurrent.futures.as_completed(future_to_info):
            idx, task = future_to_info[future]
          
            try:
                max_depth, rank, log_msgs = future.result()
                res_tuple = (max_depth, rank, log_msgs)
            except Exception as e:
                current_depth = task.get("depth", -1) if isinstance(task, dict) else -1
                err_msg = [f"[Task-{idx}] SYSTEM ERROR: {str(e)}", traceback.format_exc()]
                res_tuple = (current_depth, None, err_msg)
            
            
            pending_results[idx] = res_tuple
            results[idx] = res_tuple 

            with open(log_path, "a", encoding="utf-8") as f:
                while next_to_write in pending_results:
                    d_depth, d_rank, d_logs = pending_results.pop(next_to_write)
                   
                    overall_total += 1
                    if d_rank is not None:
                        overall_hit[d_rank:] += 1
                  
                    cur_acc = 100.0 * overall_hit / (overall_total + 1e-8)
                    now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
                    
                    rank_str = f"{d_rank+1}" if d_rank is not None else "Miss"
                    acc_str = "  ".join([f"Top-{k+1}:{cur_acc[k]:6.2f}%" for k in range(args.beam_size)])
                    
                    if d_depth is not None and d_depth != -1:
                        depth_str = f"{d_depth:5d}"
                    else:
                        depth_str = "  ERR"

                    line = f"{now}  {next_to_write+1:4d}  {depth_str}  {rank_str:>6}  {acc_str}"
                    
                    if d_logs:
                        joined_logs = "; ".join(d_logs)
                        line += f"  || LOG: {joined_logs}"

                    f.write(line + "\n")
                    
                    pbar.set_postfix_str(f"Logged:#{next_to_write+1} R={rank_str} | {acc_str[:35]}...")
                    next_to_write += 1

            if next_to_write % 10 == 0:
                torch.cuda.empty_cache()
            pbar.update(1)
        pbar.close()
    return results

def load_dataset(split, data_dir="/root/AKG/Data"):
    file_name = os.path.join(data_dir, f"{split}_dataset.json")
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return []
        
    dataset = [] 
    total_gt_sets_count = 0  #

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
                if product is None: continue 
                
                count_valid_products += 1 

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

# ==========================================
# 6. Main Execution
# ==========================================
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
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    
    parser.add_argument('--use_value', action='store_true', default=False, help='If true, use heuristic value function')
    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--value_n_layers', type=int, default=1)
    parser.add_argument('--value_latent_dim', type=int, default=128)

    args = parser.parse_args()
    
    val_tag = "UseVal" if args.use_value else "NoVal"
    log_file = f"retro_search_log_B{args.beam_size}_T{args.temperature}_P{args.length_penalty_alpha}_{val_tag}.txt"
    
    print(f"Config: Beam={args.beam_size}, Temp={args.temperature}, Alpha={args.length_penalty_alpha}, UseValue={args.use_value}")
    
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    if os.path.exists(checkpoint_path):
        print(f"Loading Transformer from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        if hasattr(checkpoint, 'state_dict'): state_dict = checkpoint.state_dict()
        predict_model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Transformer checkpoint not found at {checkpoint_path}!")

    predict_model.to(device)
    predict_model.eval()

    # load Value Model (only load weights when use_value=True, otherwise just occupy a placeholder)
    value_model = ValueMLP(
        n_layers=args.value_n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.value_latent_dim,
        dropout_rate=0.1
    )
    
    if args.use_value:
        value_ckpt_path = 'FusionRetro/value_mlp.pkl'
        if os.path.exists(value_ckpt_path):
            print(f"Loading Value Model from {value_ckpt_path}")
            value_state = torch.load(value_ckpt_path, map_location=device)
            value_model.load_state_dict(value_state)
        else:
            print("Warning: Value model checkpoint not found! (UseValue is True)")
    
    value_model.to(device)
    value_model.eval()


    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()


    print("Loading stock...")
    stock_file = '/root/AKG/Data/zinc_stock_17_04_20.hdf5'
    if os.path.exists(stock_file):
        print("Loading stock...")
        stock = pd.read_hdf(stock_file, key="table")  
        stockinchikey_list = stock.inchi_key.values
        stock_inchikeys = set([x[:14] for x in stockinchikey_list])
    else:
        print("Warning: Stock file not found.")
        stock_inchikeys = set()

    print("Initializing Searcher...")
    searcher = SimilaritySearcher(entity_file, features_path, cache_file, use_gpu=False)

    # load tasks
    print("Loading tasks...")
    tasks = load_dataset('test', data_dir="Data/Test")
    if not tasks:
        print("No tasks loaded.")
        exit()

    
    if tasks:
        results = run_parallel_with_logging(tasks, searcher, max_workers=args.workers, log_path=log_file)
        
        # summary
        overall_result = np.zeros((args.beam_size, 2))
        depth_hit = np.zeros((2, 16, args.beam_size))
        
        for i, (max_depth, rank, _) in enumerate(results):
            if max_depth is None: continue
            d_idx = min(max_depth, 15)
            overall_result[:, 1] += 1
            if rank is not None:
                overall_result[rank:, 0] += 1
            
            depth_hit[1, d_idx, :] += 1 
            if rank is not None:
                depth_hit[0, d_idx, rank:] += 1 
        
        acc_total = 100 * overall_result[:, 0] / (overall_result[:, 1] + 1e-8)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n\n====== Final Overall Result ======\n")
            print("\n====== Final Overall Result ======")
            for i in range(args.beam_size):
                msg = f"Top-{i+1} accuracy: {acc_total[i]:.2f}% ({int(overall_result[i,0])}/{int(overall_result[i,1])})"
                print(msg)
                f.write(msg + "\n")
            
            f.write("\n====== Depth-wise Result ======\n")
            for d in range(1, 16):
                total_at_depth = depth_hit[1, d, 0]
                if total_at_depth > 0:
                    accs = []
                    for k in range(args.beam_size):
                        hit = depth_hit[0, d, k]
                        acc = 100 * hit / total_at_depth
                        accs.append(f"Top-{k+1}: {acc:5.2f}%")
                    msg = f"Depth {d:2d} ({int(total_at_depth):4d}):  " + "  ".join(accs)
                    f.write(msg + "\n")

                    
