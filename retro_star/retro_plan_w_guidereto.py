#can do
# root        : INFO     Succ: 1/75/190 | avg time: 0.20 s | avg iter: 5.00
import numpy as np
import torch
import random
import logging
import time
import pickle
import sys
import os
from typing import List, Tuple
import torch
from rdkit import Chem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retro_star.alg.mol_tree import MolTree
from retro_star.common import args, prepare_starting_molecules,  \
     smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger

from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char
from Similar_search import SimilaritySearcher
from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')



_products_ids_fixed = None          # [depth, max_len]
_products_mask_fixed = None
_reactants_ids_prefix_fixed = None  # 前 k 行固定部分 [depth, max_len]
_reactants_mask_prefix_fixed = None
_memory_mask_fixed = None           # [depth]
_k = None                           # depth - 1


def _precompute_fixed_parts(products: List[str], max_depth: int, max_length: int):
    """
    在每次新分子（新的 products 列表）进入 beam search 前调用一次
    """
    global _products_ids_fixed, _products_mask_fixed
    global _reactants_ids_prefix_fixed, _reactants_mask_prefix_fixed
    global _memory_mask_fixed, _k

    depth = len(products)
    _k = depth - 1

    # ---------- products (完全不变) ----------
    _products_ids_fixed  = torch.zeros((depth, max_length), dtype=torch.long, device=device)
    _products_mask_fixed = torch.zeros((depth, max_length), device=device)

    for i, p in enumerate(products):
        seq = '^' + p + '$'
        l = min(len(seq), max_length)
        _products_ids_fixed[i, :l]  = torch.tensor([char_to_ix[c] for c in seq[:l]], device=device)
        _products_mask_fixed[i, :l] = 1

    # ---------- reactants 前 k 行 (已知等于前 k 个 product) ----------
    _reactants_ids_prefix_fixed  = torch.zeros((depth, max_length), dtype=torch.long, device=device)
    _reactants_mask_prefix_fixed = torch.zeros((depth, max_length), device=device)

    for i in range(_k):  # 0 到 k-1
        seq = '^' + products[i] + '$'
        l = len(seq) - 1
        ll = min(l, max_length)
        _reactants_ids_prefix_fixed[i, :ll]  = torch.tensor([char_to_ix[c] for c in seq[:ll]], device=device)
        _reactants_mask_prefix_fixed[i, :ll] = 1

    # memory mask
    _memory_mask_fixed = torch.ones((depth,), device=device)

# ==================== 超快批量前向（核心）====================
def get_output_probs_batch_fast(candidates: List[str], 
                                product_exter_feature,
                                max_length: int) -> torch.Tensor:
    """
    优化版：candidates 是当前正在生成的第 k 行的前缀字符串列表（不含 ^ 和 $）
    返回 [batch, vocab_size]
    """
    global _k
    batch_size = len(candidates)
    if batch_size == 0:
        return torch.empty((0, vocab_size), device=device)

    # ==== 直接复制预计算好的固定部分 ====
    products_ids   = _products_ids_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
    products_mask  = _products_mask_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
    reactants_ids  = _reactants_ids_prefix_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
    reactants_mask = _reactants_mask_prefix_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
    memory_mask    = _memory_mask_fixed.unsqueeze(0).repeat(batch_size, 1)

    # ==== 只填充第 k 行（唯一变化的部分）====
    for b, cand in enumerate(candidates):
        seq = '^' + cand                     # 输入序列：^ + prefix
        for j, c in enumerate(seq):
            if j >= max_length:
                break
            reactants_ids[b, _k, j] = char_to_ix[c]
        l = len(seq)
        if l > max_length:
            l = max_length
        reactants_mask[b, _k, :l] = 1

    # ==== mask 处理 ====
    mutual_mask    = get_mutual_mask([reactants_mask, products_mask])
    products_mask  = get_padding_mask(products_mask)
    reactants_mask = get_tril_mask(reactants_mask)
    memory_mask    = get_mem_tril_mask(memory_mask)

    # ==== 外部特征 ====
    if isinstance(product_exter_feature, np.ndarray):
        product_exter_feature = torch.from_numpy(product_exter_feature).float().to(device)
    else:
        product_exter_feature = product_exter_feature.to(device)
    expanded_features = product_exter_feature.unsqueeze(0).expand(batch_size, -1, -1)

    # ==== 前向 ====
    with torch.no_grad():
        logits = predict_model(
            products_ids, reactants_ids,
            products_mask, reactants_mask,
            mutual_mask, memory_mask,
            expanded_features
        )

    # ==== 提取概率（数值稳定）====
    probs = []
    for i, cand in enumerate(candidates):
        pos = len(cand)                                      # 关键：原始代码就是 len(prefix)
        logit = logits[i, _k, pos] / args.temperature
        prob = torch.nn.functional.softmax(logit, dim=-1)    # 最稳定方式
        probs.append(prob)

    return torch.stack(probs)  # [batch, vocab_size]
def get_beam(products,
             product_exter_feature,
             beam_size: int = 10,
             step_k: int = 8,          # 推荐 12~16，越大越准确但稍慢
             max_return: int = 10,
             alpha: float = 0.5):

    # ==================== 1. 预计算固定部分（只执行一次）===================
    _precompute_fixed_parts(products, len(products), args.max_length)

    EOS_ID = char_to_ix.get('$', -1)
    depth = len(products)
    max_len = args.max_length

    # ==================== 2. Beam 状态 ====================
    beam_tokens  = [[] for _ in range(beam_size)]               # list[token_id]
    beam_scores  = torch.zeros(beam_size, device=device)        # 累计 -log10(p)
    beam_lengths = torch.zeros(beam_size, dtype=torch.int, device=device)

    finished = []  # [(canonical_list, score)]

    # ==================== 辅助函数 ====================
    def tokens_to_string(token_list):
        return ''.join(ix_to_char[tid] for tid in token_list) if token_list else ''

    def is_valid_sequence(seq_with_dollar: str) -> bool:
        if not seq_with_dollar or seq_with_dollar == "$":
            return False
        for part in seq_with_dollar.split("."):
            smi = part.replace("$", "")
            if not smi:
                continue
            if Chem.MolFromSmiles(smi) is None:
                return False
        return True

    # ==================== 主循环 ====================
    for step in range(max_len):
        alive_idx = [i for i, tk in enumerate(beam_tokens) if tk is not None]
        if not alive_idx:
            break

        # 当前所有活跃 beam 的前缀（只生成最后一行的字符串）
        if step == 0:
            current_prefixes = [""]  # 初始空前缀
        else:
            current_prefixes = [tokens_to_string(beam_tokens[i]) for i in alive_idx]

        # 超快批量前向
        probs = get_output_probs_batch_fast(
            current_prefixes,
            product_exter_feature,
            max_length=args.max_length
        )  # [alive, vocab]

        log_prob = -torch.log10(probs + 1e-12)

        # ------------------- 长度归一化 + 局部 top-k -------------------
        if step == 0:
            k_local = min(step_k, vocab_size)
            cand_scores, cand_tokens = torch.topk(log_prob[0], k_local, largest=False)
            cand_origin = torch.zeros(k_local, dtype=torch.long, device=device)
        else:
            cur_scores = beam_scores[alive_idx]
            next_len   = beam_lengths[alive_idx] + 1
            lp         = ((5 + next_len) / 6) ** alpha
            total      = (cur_scores / lp).unsqueeze(1) + log_prob

            k_local = min(step_k, vocab_size)
            local_vals, local_idx = torch.topk(total, k_local, dim=1, largest=False)

            flat_scores = local_vals.reshape(-1)
            flat_tok    = local_idx.reshape(-1)
            flat_orig   = torch.arange(len(alive_idx), device=device).unsqueeze(1).repeat(1, k_local).reshape(-1)

            k_global = min(beam_size, flat_scores.numel())
            cand_scores, pos = torch.topk(flat_scores, k_global, largest=False)
            cand_origin = flat_orig[pos]
            cand_tokens = flat_tok[pos]

        # ------------------- 扩展候选 -------------------
        new_tokens = []
        new_scores = []
        new_lengths = []

        for i in range(cand_scores.shape[0]):
            score     = cand_scores[i].item()
            token_id  = cand_tokens[i].item()

            if step == 0:
                parent_idx = 0
            else:
                parent_alive_idx = cand_origin[i].item()
                parent_idx       = alive_idx[parent_alive_idx]

            if token_id == EOS_ID:
                completed = ("" if step == 0 else tokens_to_string(beam_tokens[parent_idx])) + '$'
                if is_valid_sequence(completed):
                    # 原始归一化方式：除以包含 $ 的总长度
                    final_score = score / len(completed)
                    reactants = [r.replace("$", "") for r in completed.split(".") if r.replace("$", "")]
                    canonical = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(r)) for r in reactants)
                    finished.append((canonical, final_score))

                    if len(finished) >= max_return:
                        step = max_len  # 触发 break
                        break
            else:
                parent = [] if step == 0 else beam_tokens[parent_idx]
                new_tokens.append(parent + [token_id])
                new_scores.append(score)
                new_lengths.append((1 if step == 0 else beam_lengths[parent_idx].item()) + 1)

        # ---------------- 更新活跃 beam ----------------
        if new_tokens:
            if len(new_tokens) > beam_size:
                # 按分数从小到大排序（-logp 越小越好）
                indices = torch.argsort(torch.tensor(new_scores))
                new_tokens   = [new_tokens[i.item()]   for i in indices[:beam_size]]
                new_scores   = [new_scores[i.item()]   for i in indices[:beam_size]]
                new_lengths  = [new_lengths[i.item()]  for i in indices[:beam_size]]

            num_new = len(new_tokens)
            pad = beam_size - num_new

            beam_tokens  = new_tokens + [None] * pad
            beam_scores  = torch.full((beam_size,), 1e9, device=device, dtype=torch.float)
            beam_lengths = torch.zeros(beam_size, dtype=torch.int, device=device)

            if num_new > 0:
                beam_scores[:num_new]  = torch.tensor(new_scores[:num_new], device=device)
                beam_lengths[:num_new] = torch.tensor(new_lengths[:num_new], device=device)
        else:
            break

        if len(finished) >= max_return:
            break

    # ------------------- 返回结果 -------------------
    finished.sort(key=lambda x: x[1])
    return [[can, score] for can, score in finished[:max_return]]


def prepare_molstar_planner(value_fn, starting_mols,
                            iterations, viz=False, viz_dir=None, searcher=None):
    # expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        # expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        searcher=searcher
    )
    return plan_handle

def molstar(target_mol, target_mol_id, starting_mols, value_fn,
            iterations, viz=False, viz_dir=None, searcher=None):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn,
        searcher=searcher
    )

    i = -1

    if not mol_tree.succ:
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            # result = expand_fn(m_next.mol)
            products=m_next.product_list
            product_exter_feature = m_next.product_exter_feature
            result = get_beam(products, product_exter_feature, beam_size=args.beamsize,
                              step_k=args.step_k, max_return=args.maxreturn, alpha=args.alpha)

            if result is not None and len(result) > 0:
                reactant_lists = []
                scores = []
                for reactants, score in result:
                    reactant_lists.append(reactants)
                    # 保证 score 是 float
                    scores.append(score.cpu().item() if torch.is_tensor(score) else float(score))
                costs = np.array(scores)
                templates = [''] * len(reactant_lists)  # 如果没有模板信息，可设为None

                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs, templates)

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, i+1)



def retro_plan():

    starting_mols = prepare_starting_molecules(args.starting_molecules)

    routes = pickle.load(open(args.test_routes, 'rb'))
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.test_routes))
    # # # ========== 新增这几行 ==========
    ### 1.6
    # 210
    # 81,107,143,149,264,295,322,359
    ### 1.5
    # 405,699,730
    # INDICES_TO_TEST = [81,98,143,149,264,270,295,322,324,359,409,432,489,604,662,702,767,774,795,903,964,971]  
    # routes = [routes[i] for i in INDICES_TO_TEST]
    # logging.info(f"【调试模式】只运行指定的 {len(routes)} 个分子，索引: {INDICES_TO_TEST}")
    
    # # ====================================

    # one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump)

    # create result folder
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.use_value_fn:
        model = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f,  map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    plan_handle = prepare_molstar_planner(
    
        value_fn=value_fn,
        starting_mols=starting_mols,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir,
        searcher=searcher
    )

    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': []
    }
    num_targets = len(routes)
    t0 = time.time()
    for i, route in tqdm(enumerate(routes), total=num_targets, desc="Planning"):
        # if i !=74:
        #     continue
        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(target_mol, i)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))
        # if i==74:
        #     raise KeyboardInterrupt
    if args.use_value_fn:
        with open(args.result_folder + f'/plan_chembl_w_ours_useV_{args.temperature}.pkl', 'wb') as f:
            pickle.dump(result, f)
    else:
        with open(args.result_folder + f'/plan_chembl_w_ours_noV_{args.temperature}.pkl', 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    if args.use_value_fn:
        setup_logger(f'plan_chembl_w_ours_useV_{args.temperature}.log')
    else:
        setup_logger(f'plan_chembl_w_ours_noV_{args.temperature}.log')
    logging.info('seed: %d' % args.seed)
    logging.info('temperature: %.2f' % args.temperature)
    logging.info("iterations: %d" % args.iterations)
    logging.info("beamsize: %d" % args.beamsize)
    logging.info("step_k: %d" % args.step_k)
    logging.info("maxreturn: %d" % args.maxreturn)
    logging.info("alpha: %.2f" % args.alpha)
    logging.info('test_routes: %s' % args.test_routes)
    
    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    entity_file = 'Data/Train/for_embedding/all_molecules_clean.txt'
    features_path = 'rgcn/global_emb_FP_512/model_epoch197_0.00307_embedding.npy'
    ckpt_dir = "FusionRetro/models_final_FP512_Noclip_moresize"
    os.makedirs(ckpt_dir, exist_ok=True)
    cache_file = f"{ckpt_dir}/fp_cache.npz"
    checkpoint_path = f"{ckpt_dir}/finetune_best_model_ffn_gate_path_6249_2728.pth"
    searcher = SimilaritySearcher(entity_file, features_path, cache_file, use_gpu=False)
    logging.info("Similarity Searcher initialized.")


    config = TransformerConfig(vocab_size=get_vocab_size(),
                            max_length=args.max_length,
                            embedding_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=args.num_attention_heads,
                            intermediate_size=args.intermediate_size,
                            hidden_dropout_prob=args.hidden_dropout_prob)

    predict_model = Transformer(config)
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    predict_model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    
    predict_model.to(device)
    predict_model.eval()
    logging.info('Transformer model loaded from %s' % checkpoint_path)


    retro_plan()

# python retro_star/retro_plan_w_trans.py --use_value_fn --temperature 0.9 --beamsize 10 --step_k 8 --maxreturn 10 --alpha 0.5 --iterations 50